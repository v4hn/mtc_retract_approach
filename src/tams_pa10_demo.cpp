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

#include <shape_msgs/Mesh.h>

// for createQuaternionMsgFromRollPitchYaw
#include <tf/transform_datatypes.h>

#define M_TAU (2. * M_PI)

using namespace moveit::task_constructor;

static Task *active_task{nullptr};
void sig_handler(int s)
{
    if (active_task)
        active_task->preempt();
    ros::requestShutdown();
}

void collisionObjectFromResource(moveit_msgs::CollisionObject &msg,
                                 const std::string &id,
                                 const std::string &resource)
{
    msg.meshes.resize(1);

    // load mesh
    const Eigen::Vector3d scaling(1, 1, 1);
    shapes::Shape *shape = shapes::createMeshFromResource(resource, scaling);
    shapes::ShapeMsg shape_msg;
    shapes::constructMsgFromShape(shape, shape_msg);
    msg.meshes[0] = boost::get<shape_msgs::Mesh>(shape_msg);

    // set pose
    msg.mesh_poses.resize(1);
    msg.mesh_poses[0].orientation.w = 1.0;

    // fill in details for MoveIt
    msg.id = id;
    msg.operation = moveit_msgs::CollisionObject::ADD;
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

    auto bottle = []
    {
        moveit_msgs::CollisionObject obj;
        collisionObjectFromResource(obj, "bottle", "package://mtc_retract_approach/meshes/bottle_tall.stl");
        obj.header.frame_id = "table_coordinate_grid";
        obj.pose.position.x = 0.0;
        obj.pose.position.y = 0.0;
        obj.pose.position.z = 0.0;
        obj.pose.orientation.w = 1.0;
        return obj;
    }();

    auto bottle_padded = []
    {
        moveit_msgs::CollisionObject obj;
        obj.id = "bottle_padded";
        obj.primitives.resize(1);
        obj.primitives[0].type = shape_msgs::SolidPrimitive::CYLINDER;
        obj.primitives[0].dimensions = {0.4, 0.09};
        obj.header.frame_id = "table_coordinate_grid";
        obj.pose.position.x = 0.0;
        obj.pose.position.y = 0.0;
        obj.pose.position.z = obj.primitives[0].dimensions[0] / 2;
        obj.pose.orientation.w = 1.0;
        return obj;
    }();

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
        auto fixed = std::make_unique<stages::FixedState>("fixed");
        // get Scene from a new monitor
        auto scene = std::make_shared<planning_scene::PlanningScene>(t.getRobotModel());
        scene->processCollisionObjectMsg(bottle);
        scene->getCurrentStateNonConst().setToDefaultValues("qbsc_gripper_group", "fully_open");
        fixed->setState(scene);
        auto _first = std::make_unique<stages::ComputeIK>("first", std::move(fixed));
        _first->setTargetPose(
            []
            {
                geometry_msgs::PoseStamped p;
                p.header.frame_id = "table_coordinate_grid";
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
        // todo: add pose property
        first = _first.get();
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
        stage->setMinMaxDistance(.1, .1);
        t.add(std::move(stage));
    }

    {
        auto stage = std::make_unique<stages::ModifyPlanningScene>("add padding");
        stage->addObject(bottle_padded);
        t.add(std::move(stage));
    }

    {
        auto stage = std::make_unique<stages::Connect>(
            "move to pre-grasp pose",
            stages::Connect::GroupPlannerVector{{"pa10_opw_group", sampling_planner}});
        stage->properties().configureInitFrom(Stage::PARENT);
        stage->setComputeAttempts(connect_compute_attempts);
        t.add(std::move(stage));
    }

    {
        auto stage = std::make_unique<stages::ModifyPlanningScene>("remove padding");
        auto bottle_padded_rm{ bottle_padded };
        bottle_padded_rm.operation = moveit_msgs::CollisionObject::REMOVE;
        stage->removeObject(bottle_padded_rm);
        stage->setMaxSolutions(1); // TODO: max_solutions does not work here
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
        stage->setMinMaxDistance(.1, .1);

        t.add(std::move(stage));
    }

    {
        auto _s = std::make_unique<stages::GeneratePose>("fixed");
        _s->setPose(
            []
            {
                geometry_msgs::PoseStamped p;
                p.header.frame_id = "table_coordinate_grid";
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

    t.stages()->setCostTerm(std::make_shared<cost::TrajectoryDuration>());

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
        t.plan(pnh.param<int>("solutions", 0));
        ROS_WARN_STREAM("Planning took "
                        << (ros::WallTime::now() - start_time).toSec() * 1000.0
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

    // If wanted, keep introspection alive
    if (pnh.param("keep_running", true))
    {
        t.introspection();
        ros::waitForShutdown();
    }

    return 0;
}
