# Nominal source world

All positions and dimensions below are metres in the world frame. Quaternions are
scalar-first. These describe the nominal source configuration, including authored
asset data and runtime overrides; they are not live simulator measurements.

## Bodies and geometry

| Body | Placement and geometry | Dynamics |
|---|---|---|
| Red cube | centre [0.2, -0.13, 0.7775], quaternion [1,0,0,0], scale [1,1,1]; cube side 0.025 | rigid, mass 0.35 kg; initial linear/angular velocity [0,0,0] |
| Plate | centre [-0.02,-0.08,0.7828], quaternion [1,0,0,0], scale [1,1,1]; imported plate mesh | configured mass 0.5 kg; made kinematic by task; zero-velocity initialization disabled on retained resets |
| Table | static cuboid dimensions [1.4,1.1,0.05], centre [0,-0.05,0.74], quaternion [1,0,0,0]; top z=0.765 | immovable collider; convex-decomposition approximation requested |
| Ground | cube geometry, thickness 0.1, Z axis, position [0,0,0] | ground collider |
| Room | Simple_Room_nolight, position [0,0,0], quaternion [1,0,0,0], scale [0.5,0.5,0.5] | collision enabled by room constructor default |

The plate metadata aligned bounds are [-0.0648,-0.0647,-0.0178] to
[0.0647,0.0648,0.0178]. The cube asset has 8 vertices and 12 faces; the plate
metadata lists 140013 vertices and 228114 faces. Exact collision meshes and convex
decomposition settings beyond those stated here are not specified in the source
configuration. Cube/plate inertia tensors, centre-of-mass overrides, and room/ground
masses are not specified in the source configuration.

Both objects request a stability check. Layouts displace the cube uniformly by up
to 0.03 in each horizontal axis. Plate displacement is uniform in x=[0,0.03],
y=[-0.03,0]. No initial orientation randomization is applied.

## Materials

| Material/application | Static friction | Dynamic friction | Restitution | Combination/notes |
|---|---|---|---|---|
| Cube effective authored task material | 1.0 | 1.0 | 0.0 | friction multiply; bound before native shape construction, stronger than descendants |
| Cube scene declaration | 0.8 | 0.6 | 0.0 | superseded by task material above |
| Plate scene material | 0.8 | 0.6 | 0.0 | native material binding; no task combine override |
| Table material declaration | 0.8 | 0.8 | 0 | constructed by table code; static branch does not explicitly bind it |
| Scene default rigid-body material | 0.5 | 0.5 | 0.0 | friction average, restitution average |
| Ground material | 1.0 | 1.0 | 0.0 | friction multiply, restitution multiply; compliant stiffness/damping 0/0 |

The cube task material replaces the scene declaration; material values from the
asset metadata are not applied by that task override. No explicit finger, other
robot-link, or room material coefficients are specified in the source configuration;
shape-level bindings in imported binary assets are not established by the text
configuration. Cube restitution-combine mode and plate/table restitution/friction
combine overrides are not specified in the source configuration. Default material
compliance stiffness/damping are 0/0.

Visual materials: table material_0122, ground material_0564 (MDL, random false),
room Simple_Room_nolight, background brown_photostudio_02_4k.hdr with intensity
1000. Cube and plate have empty scene visual overrides and use their assets.
