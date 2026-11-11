# Robot and control

Two uncoupled X5 arms have fixed roots at [-0.3,-0.45,0.765] and
[0.3,-0.45,0.765], each with quaternion [0.707,0,0,0.707]. Each has six
revolute arm joints (joint1–joint6), two prismatic fingers (joint7, joint8),
and a wrist camera. The right arm acts; the left arm retains its command.
The end-effector link is link6, base is base_link. Finger joint7 is the command
base; joint8 follows with multiplier 1 and offset 0. Gripper bias is 0.145 m;
joint endpoints are [-0.01,0.044] m. The fixed task gripper multiplier is 1.0.

Initial joint1–joint6 positions are zero; joint7/joint8 are 0.044. The asset config
default root [0.25,-0.25,0] is replaced by the two scene roots above. Robot gravity
is disabled, maximum depenetration velocity is 5 m/s, self-collisions are enabled,
root is fixed, position solver iterations are 8 and velocity iterations are 0.

| Drive | Joints | Stiffness | Damping | Effort limit | Velocity limit | Armature |
|---|---|---|---|---|---|---|
| Implicit arm | joint1–joint6 | 4400 | 40 | 100 | 5 rad/s | 0.01 |
| Implicit gripper | joint7, joint8 | 2300 | 100 | 100 | not specified in the source configuration | not specified in the source configuration |

Arm effort is torque (N m); finger effort is force (N). Stiffness/damping use the
corresponding rotational or translational SI units. Nonzero right squeeze commands
set finger stiffness to zero and effort targets to minus the commanded force on
both fingers. Zero force restores nominal stiffness and zero effort targets.
Damping remains as configured. Nominal stiffness and limits are retained across
episode resets. The opening controller advances toward its target with 0.2 of
the joint endpoint span per command until within that fraction.

The five policy outputs clip to [-1,1]. First three integrate a pad-centre command
by 0.01 m per step per axis; the pad floor is z=0.778. End-effector xyz clips to
[-0.10,-0.45,0.85] through [0.40,0.10,1.20]. A fixed top-down orientation is
approached over 40 steps with quintic interpolation. Output four closes when
positive, otherwise opens. Output five maps to 25+20*a N when closed and 0 N
when open. The batch backend uses the X5 URDF forward kinematics and Jacobian
with damped-least-squares IK, seeded from current arm joints. A failed solve
retains the previous arm joint target. Successful arm targets interpolate in
joint space over the first eight physics substeps (fractions 1/9 through 8/9),
then hold for two substeps. Gripper position commands follow the same profile
and the opening controller described above. Robot meshes supply visual and
collision geometry; mesh coordinates and imported USD shape material bindings
are not specified in the source configuration described by these tables.

The accompanying robot_asset.md lists each nominal URDF link, inertial frame,
joint transform and limit. These are asset source values; actuator limits above
override the URDF effort/velocity declarations. A generated USD is the loaded
robot asset; these tables do not assert independently measured live inertias.
