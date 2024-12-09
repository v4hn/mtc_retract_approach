#include <ros/ros.h>
#include <signal.h>

#include <moveit/robot_model/robot_model.h>
#include <moveit/planning_scene/planning_scene.h>

#include <moveit/task_constructor/task.h>

#include <moveit/task_constructor/stages/compute_ik.h>
#include <moveit/task_constructor/stages/connect.h>
#include <moveit/task_constructor/stages/fixed_state.h>
#include <moveit/task_constructor/stages/move_relative.h>
#include <moveit/task_constructor/stages/move_to.h>

#include <moveit/task_constructor/stages/modify_planning_scene.h>

#include <moveit/task_constructor/stages/generate_grasp_pose.h>
#include <moveit/task_constructor/stages/generate_place_pose.h>
#include <moveit/task_constructor/stages/generate_pose.h>

#include <moveit/task_constructor/solvers/cartesian_path.h>
#include <moveit/task_constructor/solvers/pipeline_planner.h>
#include <moveit/task_constructor/solvers/joint_interpolation.h>

#include <moveit/task_constructor/cost_terms.h>

#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_extents.h>
#include <geometric_shapes/shape_operations.h>
#include <geometric_shapes/bodies.h>

#include <shape_msgs/Mesh.h>

#include <tf/transform_datatypes.h>
#include <eigen_conversions/eigen_msg.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <optional>

#define M_TAU (2. * M_PI)

using namespace moveit::task_constructor;

static Task *active_task{nullptr};
void sig_handler(int s)
{
    if (active_task)
        active_task->preempt();
    ros::requestShutdown();
}

double computeMeshHeight(const shape_msgs::Mesh &mesh)
{
    double x, y, z;
    geometric_shapes::getShapeExtents(mesh, x, y, z);
    return z;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mtc_tutorial");
    signal(SIGINT, sig_handler);

    ros::AsyncSpinner spinner(2);
    ros::NodeHandle pnh("~");
    spinner.start();

    bodies::BoundingCylinder bottle_cylinder;
    auto bottle = [&bottle_cylinder]
    {
        moveit_msgs::CollisionObject obj;
        {
            Eigen::Vector3d const scaling(1, 1, 1);
            shapes::Shape *shape = shapes::createMeshFromResource("package://mtc_retract_approach/meshes/bottle_tall.stl", scaling);
            bodies::ConvexMesh(shape).computeBoundingCylinder(bottle_cylinder);
            shapes::ShapeMsg shape_msg;
            shapes::constructMsgFromShape(shape, shape_msg);
            obj.meshes.push_back(boost::get<shape_msgs::Mesh>(shape_msg));
            obj.mesh_poses.resize(1);
            obj.mesh_poses[0].orientation.w = 1.0;
            delete shape;
        }
        obj.id = "bottle";
        obj.operation = moveit_msgs::CollisionObject::ADD;
        obj.header.frame_id = "table_coordinate_grid";
        obj.pose.position.x = 0.1;
        obj.pose.position.y = 0.03;
        obj.pose.position.z = 0.0;
        obj.pose.orientation.w = 1.0;
        return obj;
    }();
    auto bottle_rm = [&bottle]
    {
        auto b{bottle};
        b.operation = moveit_msgs::CollisionObject::REMOVE;
        return b;
    }();

    bottle_cylinder.radius += 0.05;
    bottle_cylinder.length += 0.05;

    auto bottle_padded = [&bottle_cylinder]
    {
        moveit_msgs::CollisionObject obj;
        obj.id = "bottle_padded";
        obj.primitives.resize(1);
        obj.primitives[0].type = shape_msgs::SolidPrimitive::CYLINDER;
        obj.primitives[0].dimensions = {bottle_cylinder.length, bottle_cylinder.radius};
        obj.header.frame_id = "bottle";
        tf::poseEigenToMsg(bottle_cylinder.pose, obj.pose);

        return obj;
    }();

    auto bottle_padded_rm = [&bottle_padded]
    {
        auto b{bottle_padded};
        b.operation = moveit_msgs::CollisionObject::REMOVE;
        return b;
    }();

    bool padded = pnh.param<bool>("padded", false);

    Task t("my_task");
    t.loadRobotModel();

    int workers = pnh.param<int>("workers", -1);
    if (workers >= 0)
    {
        ROS_INFO_STREAM("Setting " << workers << " worker threads");
        t.setParallelExecutor(workers);
    }
    else
    {
        ROS_INFO("Using direct executor");
        t.setDirectExecutor();
        workers = 1;
    }

    int connect_compute_attempts = pnh.param<int>("connect_compute_attempts", 1);
    ROS_INFO_STREAM("Using " << connect_compute_attempts << " compute attempts in Connect");
    if (connect_compute_attempts < 1)
    {
        ROS_ERROR("Invalid value for 'connect_compute_attempts', must be at least 1. will assume 1 instead.");
        connect_compute_attempts = 1;
    }

    auto joint_interpolation = std::make_shared<solvers::JointInterpolationPlanner>();

    solvers::PlannerInterfacePtr sampling_planner;
    sampling_planner = std::make_shared<solvers::PipelinePlanner>();
    sampling_planner->setProperty("goal_joint_tolerance", 1e-5);

    std::chrono::duration<double> connect_timeout(1.0);

    auto cartesian_planner = std::make_shared<solvers::CartesianPath>();
    cartesian_planner->setMaxVelocityScalingFactor(1);
    cartesian_planner->setMaxAccelerationScalingFactor(1);
    cartesian_planner->setStepSize(.002);

    t.setProperty("group", "pa10_opw_group");

    Stage *first = nullptr;
    {
        auto scene = std::make_shared<planning_scene::PlanningScene>(t.getRobotModel());
        scene->processCollisionObjectMsg(bottle);
        scene->setObjectColor(
            "bottle_padded",
            []
            {
                std_msgs::ColorRGBA color;
                color.r = 0.0;
                color.g = 0.0;
                color.b = 1.0;
                color.a = 0.5;
                return color;
            }());
        scene->getCurrentStateNonConst().setToDefaultValues("qbsc_gripper_group", "fully_open");
        auto fixed = std::make_unique<stages::FixedState>("fixed");
        fixed->setState(scene);
        first = fixed.get();

        auto _first = std::make_unique<stages::ComputeIK>("first", std::move(fixed));
        _first->setTargetPose(
            []
            {
                geometry_msgs::PoseStamped p;
                p.header.frame_id = "bottle";
                p.pose.position.x = 0.0;
                p.pose.position.y = .01;
                p.pose.position.z = 0.13;
                p.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(M_TAU / 4 + M_TAU / 12, 0, M_TAU / 2);
                return p;
            }());
        _first->setIKFrame(
            []
            {
                geometry_msgs::PoseStamped p;
                p.header.frame_id = "qbsc_gripper/tcp_static";
                p.pose.position.x = -.03;
                p.pose.orientation.w = 1.0;
                return p;
            }());
        _first->properties().configureInitFrom(Stage::PARENT, {"group"});

        _first->setMaxIKSolutions(32);
        t.add(std::move(_first));
    }

    {
        auto stage = std::make_unique<stages::MoveRelative>("retract", cartesian_planner);
        stage->properties().configureInitFrom(Stage::PARENT, {"group"});
        stage->setIKFrame("qbsc_gripper/tcp_static");
        geometry_msgs::Vector3Stamped vec;
        vec.header.frame_id = "qbsc_gripper/tcp_static";
        vec.vector.z = -1.0;
        stage->setDirection(vec);
        stage->setMinMaxDistance(.11, .11);
        t.add(std::move(stage));
    }

    if (padded)
    {
        auto stage = std::make_unique<stages::ModifyPlanningScene>("add padding");
        stage->addObject(bottle_padded);
        stage->allowCollisions("bottle_padded", "table_plate", true);
        stage->removeObject(bottle_rm);
        t.add(std::move(stage));
    }

    {
        auto stage = std::make_unique<stages::Connect>(
            "transit",
            stages::Connect::GroupPlannerVector{{"pa10_opw_group", sampling_planner}});
        stage->properties().declare<std::string>("group", "group name used for clearance");
        stage->properties().configureInitFrom(Stage::PARENT);
        stage->setComputeAttempts(connect_compute_attempts);
        stage->setCostTerm(std::make_shared<cost::Clearance>());
        t.add(std::move(stage));
    }

    if (padded)
    {
        auto stage = std::make_unique<stages::ModifyPlanningScene>("remove padding");
        stage->allowCollisions("bottle_padded", "table_plate", false);
        stage->removeObject(bottle_padded_rm);
        stage->addObject(bottle);
        t.add(std::move(stage));
    }

    {
        auto stage = std::make_unique<stages::MoveRelative>("approach", cartesian_planner);
        stage->setIKFrame("qbsc_gripper/tcp_static");
        stage->properties().configureInitFrom(Stage::PARENT, {"group"});
        stage->setDirection(
            []
            {
                geometry_msgs::Vector3Stamped vec;
                vec.header.frame_id = "qbsc_gripper/tcp_static";
                vec.vector.z = 1.0;
                return vec;
            }());
        stage->setMinMaxDistance(.11, .11);

        t.add(std::move(stage));
    }

    {
        auto _s = std::make_unique<stages::GeneratePose>("fixed");
        _s->setPose(
            []
            {
                geometry_msgs::PoseStamped p;
                p.header.frame_id = "bottle";
                p.pose.position.x = 0.0;
                p.pose.position.y = -.01;
                p.pose.position.z = 0.13;
                p.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(-M_TAU / 4 - M_TAU / 12, 0, M_TAU / 2);
                return p;
            }());
        _s->setMonitoredStage(first);

        auto last = std::make_unique<stages::ComputeIK>("last", std::move(_s));
        last->properties().configureInitFrom(Stage::PARENT, {"group"});
        last->properties().configureInitFrom(Stage::INTERFACE, {"target_pose"});
        last->setIKFrame(
            []
            {
                geometry_msgs::PoseStamped p;
                p.header.frame_id = "qbsc_gripper/tcp_static";
                p.pose.position.x = -.03;
                p.pose.orientation.w = 1.0;
                return p;
            }());

        last->setMaxIKSolutions(32);
        t.add(std::move(last));
    }

    auto cost{ pnh.param<std::string>("cost", "duration") };
    if(cost == "duration")
    {
        t.stages()->setCostTerm(std::make_shared<cost::TrajectoryDuration>());
    }
    else if (cost== "pathlength") {
        t.stages()->setCostTerm(std::make_shared<cost::PathLength>());
    }
    else if (cost == "eefpath") {
        t.stages()->setCostTerm(std::make_shared<cost::LinkMotion>("qbsc_gripper/tcp_static"));
    }
    else
    {
        ROS_ERROR_STREAM("Unknown cost term '" << cost << "'");
        return 1;
    }

    if (!pnh.param<bool>("introspection", true))
        t.enableIntrospection(false);

    bool execute = pnh.param<bool>("execute", false);

    if (execute)
    {
        ROS_INFO("Going to execute best solution");
    }

    ROS_INFO_STREAM(t);

    active_task = &t;
    try
    {
        auto start_time{ros::WallTime::now()};
        ros::WallTime end_time;
        t.addSolutionCallback([&end_time](const SolutionBase &s)
                              {
                                if(end_time.isZero())
                                    end_time = ros::WallTime::now();
                              });
        t.plan(pnh.param<int>("solutions", 0));
        assert(!end_time.isZero());
        ROS_WARN_STREAM("Planning took "
                        << (end_time - start_time).toSec() * 1000.0
                        << "ms to find "
                        << t.numSolutions()
                        << " solution(s) with best solution "
                        << (t.solutions().empty() ? std::numeric_limits<double>::quiet_NaN() : t.solutions().front()->cost()));
    }
    catch (InitStageException &e)
    {
        ROS_ERROR_STREAM(e);
        return 1;
    }
    catch (std::exception &e)
    {
        ROS_ERROR_STREAM(e.what());
        return 1;
    }
    active_task = nullptr;

    if (t.numSolutions() == 0)
    {
        ROS_ERROR("No solution found.");
    }
    else if (execute)
    {
        t.execute(*t.solutions().front());
    }

    if (pnh.param<bool>("cost_by_start_end", false))
    {
        for (auto const &s : t.solutions())
        {
            auto &seq = dynamic_cast<SolutionSequence const &>(*s);
            auto *start = seq.internalStart();
            auto *end = seq.internalEnd();
            auto getString = [](auto const *s)
            {
                std::vector<double> v;
                s->scene()->getCurrentState().copyJointGroupPositions("pa10_opw_group", v);
                return fmt::format("{::.3f}", v);
            };
            ROS_INFO_STREAM(fmt::format("Cost for solution from '{}' to '{}' is {:.5f}",
                                        getString(start),
                                        getString(end),
                                        s->cost()));
        }
    }

    if (pnh.param<bool>("distance_above_object", false))
    {
        double max_height{ bottle_cylinder.length };

        for (auto &s : t.solutions())
        {
            auto &seq = dynamic_cast<SolutionSequence const &>(*s);
            double distance{0.0};
            std::optional<Eigen::Vector3d> last_position;
            for (auto &s : seq.solutions())
            {
                try
                {
                    auto &straj = dynamic_cast<SubTrajectory const &>(*s);
                    auto traj_ptr = straj.trajectory();
                    if (!traj_ptr)
                        continue;
                    auto& traj = const_cast<robot_trajectory::RobotTrajectory&>(*traj_ptr);
                    for (size_t i = 0; i < traj.getWayPointCount(); ++i)
                    {
                        traj.getWayPointPtr(i)->update();
                        auto &wp = traj.getWayPoint(i);
                        // transform to table frame, not the global one
                        auto &this_position =
                            (wp.getGlobalLinkTransform("table_coordinate_grid").inverse() * wp.getGlobalLinkTransform("qbsc_gripper/tcp_static")).translation();
                        if (!last_position)
                        {
                            last_position = this_position;
                            continue;
                        }

                        if (this_position.z() > max_height && last_position->z() > max_height)
                        {
                            distance += (this_position - *last_position).norm();
                        }
                        else if (this_position.z() > max_height)
                        {
                            auto t = (max_height - last_position->z()) / (this_position.z() - last_position->z());
                            auto p = *last_position + t * (this_position - *last_position);
                            distance += (this_position - p).norm();
                        }
                        else if (last_position->z() > max_height)
                        {
                            auto t = (max_height - last_position->z()) / (this_position.z() - last_position->z());
                            auto p = *last_position + t * (this_position - *last_position);
                            distance += (p - *last_position).norm();
                        }
                        last_position = this_position;
                    }
                }
                catch (std::bad_cast &e)
                {
                    continue;
                }
            }
            auto *start = seq.internalStart();
            auto *end = seq.internalEnd();
            auto getString = [](auto const *s)
            {
                std::vector<double> v;
                s->scene()->getCurrentState().copyJointGroupPositions("pa10_opw_group", v);
                return fmt::format("{::.3f}", v);
            };
            ROS_INFO_STREAM(fmt::format("Cost for solution from '{}' to '{}' is {:.5f}", getString(start),
                                        getString(end),
                                        distance));
        }
    }

    if (pnh.param<bool>("transit_clearance", false))
    {
        // get all solutions of "transit" stage and compute the minimum clearance for each solution
        for (auto const &s : t.stages()->findChild("transit")->solutions())
        {
            if (s->isFailure())
                continue;

            auto &straj = dynamic_cast<SubTrajectory const &>(*s);
            auto &traj = *straj.trajectory();
            auto &scene = *straj.start()->scene();

            auto check_scene = scene.diff();
            if (padded)
            {
                check_scene->processCollisionObjectMsg(bottle_padded_rm);
                check_scene->processCollisionObjectMsg(bottle);
            }
            collision_detection::DistanceRequest request;
            request.type = collision_detection::DistanceRequestType::GLOBAL;
            request.group_name = "pa10_opw_group";
            request.enableGroup(scene.getRobotModel());
            collision_detection::DistanceResult result;
            for (size_t i = 0; i < traj.getWayPointCount(); ++i)
            {
                collision_detection::DistanceResult res;
                check_scene->getCollisionEnv()->distanceRobot(request, res, traj.getWayPoint(i));
                if (res.minimum_distance.distance < result.minimum_distance.distance)
                {
                    result = res;
                }
            }
            // print resulting distance and link pairs
            ROS_INFO_STREAM(fmt::format("Minimum distance: {:.5f} between '{}' and '{}'",
                                        result.minimum_distance.distance,
                                        result.minimum_distance.link_names[0],
                                        result.minimum_distance.link_names[1]));
        }
    }

    // If wanted, keep introspection alive
    if (pnh.param("keep_running", true))
    {
        t.introspection();
        ros::waitForShutdown();
    }

    return 0;
}
