# Nominal robot asset tables

Positions and centres of mass are metres, angles radians, masses kg, inertias kg m².
Origins use xyz and rpy in the parent/link frame. These are nominal URDF source
declarations, not measurements of the loaded USD. Mesh geometry uses its authored
coordinates. Unlisted properties are not specified in the source configuration.

## Bodies

| Link | Mass | Inertial origin | Inertia |
|---|---|---|---|
| base_link | value=0.43995 | xyz=-3.0001E-05 3.0589E-05 0.027324, rpy=0 0 0 | ixx=0.00017, ixy=0.00000, ixz=0.00000, iyy=0.00017, iyz=0.00000, izz=0.00016 |
| link1 | value=0.066982 | xyz=0.0054231 -0.0080289 0.017086, rpy=0 0 0 | ixx=0.00008, ixy=0.00000, ixz=0.00001, iyy=0.00003, iyz=0.00000, izz=0.00008 |
| link2 | value=1.0795 | xyz=-0.13237 0.0020852 0.00010549, rpy=0 0 0 | ixx=0.00051, ixy=0.00001, ixz=-0.00004, iyy=0.01599, iyz=0.00000, izz=0.01605 |
| link3 | value=0.54534 | xyz=0.18531 0.00068376 -0.051638, rpy=0 0 0 | ixx=0.00036, ixy=-0.00003, ixz=-0.00039, iyy=0.00423, iyz=0.00001, izz=0.00420 |
| link4 | value=0.11714 | xyz=0.040231 0.0044807 -0.035335, rpy=0 0 0 | ixx=0.00019, ixy=-0.00002, ixz=-0.00009, iyy=0.00023, iyz=0.00002, izz=0.00014 |
| link5 | value=0.63488 | xyz=0.003612 -1.5455E-05 0.055214, rpy=0 0 0 | ixx=0.00083, ixy=0.00000, ixz=0.00007, iyy=0.00082, iyz=0.00000, izz=0.00026 |
| link6 | value=0.44089 | xyz=0.041697 2.4368E-05 0.00014464, rpy=0 0 0 | ixx=0.00038, ixy=0.00000, ixz=0.00000, iyy=0.00028, iyz=0.00000, izz=0.00050 |
| camera_base | not specified in the source configuration | not specified in the source configuration | not specified in the source configuration |
| camera | not specified in the source configuration | not specified in the source configuration | not specified in the source configuration |
| link7 | value=0.064798 | xyz=-0.00035522 -0.007827 -0.0029883, rpy=0 0 0 | ixx=0.00002, ixy=0.00000, ixz=0.00000, iyy=0.00003, iyz=0.00000, izz=0.00003 |
| link8 | value=0.064798 | xyz=-0.00035522 0.0078277 0.0024201, rpy=0 0 0 | ixx=0.00002, ixy=0.00000, ixz=0.00000, iyy=0.00003, iyz=0.00000, izz=0.00003 |

## Geometry and visual materials

| Link | Role | Geometry | Origin | Visual material/color |
|---|---|---|---|---|
| base_link | visual | mesh filename=meshes/base_link.STL | xyz=0 0 0, rpy=0 0 0 | name=black; rgba=0 0 0 1 |
| base_link | collision | mesh filename=meshes/base_link.STL | xyz=0 0 0, rpy=0 0 0 | not specified in the source configuration; not specified in the source configuration |
| link1 | visual | mesh filename=meshes/link1.STL | xyz=0 0 0, rpy=0 0 0 | name=black; rgba=0 0 0 1 |
| link1 | collision | mesh filename=meshes/link1.STL | xyz=0 0 0, rpy=0 0 0 | not specified in the source configuration; not specified in the source configuration |
| link2 | visual | mesh filename=meshes/link2.STL | xyz=0 0 0, rpy=0 0 0 | name=black; rgba=0 0 0 1 |
| link2 | collision | mesh filename=meshes/link2.STL | xyz=0 0 0, rpy=0 0 0 | not specified in the source configuration; not specified in the source configuration |
| link3 | visual | mesh filename=meshes/link3.STL | xyz=0 0 0, rpy=0 0 0 | name=white; rgba=1 1 1 1 |
| link3 | collision | mesh filename=meshes/link3.STL | xyz=0 0 0, rpy=0 0 0 | not specified in the source configuration; not specified in the source configuration |
| link4 | visual | mesh filename=meshes/link4.STL | xyz=0 0 0, rpy=0 0 0 | name=black; rgba=0 0 0 1 |
| link4 | collision | mesh filename=meshes/link4.STL | xyz=0 0 0, rpy=0 0 0 | not specified in the source configuration; not specified in the source configuration |
| link5 | visual | mesh filename=meshes/link5.STL | xyz=0 0 0, rpy=0 0 0 | name=black; rgba=0 0 0 1 |
| link5 | collision | mesh filename=meshes/link5.STL | xyz=0 0 0, rpy=0 0 0 | not specified in the source configuration; not specified in the source configuration |
| link6 | visual | mesh filename=meshes/link6.STL | xyz=0 0 0, rpy=0 0 0 | name=black; rgba=0 0 0 1 |
| link6 | collision | mesh filename=meshes/link6.STL | xyz=0 0 0, rpy=0 0 0 | not specified in the source configuration; not specified in the source configuration |
| camera_base | visual | mesh filename=meshes/camera_base.glb | xyz=0 0 0.0, rpy=0 0 0 | not specified in the source configuration; not specified in the source configuration |
| camera_base | collision | not specified in the source configuration | not specified in the source configuration | not specified in the source configuration |
| camera | visual | mesh filename=meshes/camera.glb | rpy=0.0 0.0 -1.5707963267948966, xyz=0.0 0.0 -0.0 | not specified in the source configuration; not specified in the source configuration |
| camera | collision | not specified in the source configuration | not specified in the source configuration | not specified in the source configuration |
| link7 | visual | mesh filename=meshes/link7.STL | xyz=0 0 0, rpy=0 0 0 | name=black; rgba=0 0 0 1 |
| link7 | collision | mesh filename=meshes/link7.STL | xyz=0 0 0, rpy=0 0 0 | not specified in the source configuration; not specified in the source configuration |
| link8 | visual | mesh filename=meshes/link8.STL | xyz=0 0 0, rpy=0 0 0 | name=black; rgba=0 0 0 1 |
| link8 | collision | mesh filename=meshes/link8.STL | xyz=0 0 0, rpy=0 0 0 | not specified in the source configuration; not specified in the source configuration |

## Joints

| Joint/type | Parent → child | Origin | Axis | Limits |
|---|---|---|---|---|
| joint1 / revolute | base_link → link1 | xyz=0 0 0.0605, rpy=0 0 0 | xyz=0 0 1 | lower=-10, upper=10, effort=100, velocity=1000 |
| joint2 / revolute | link1 → link2 | xyz=0.02 0 0.04, rpy=0 0 0 | xyz=0 1 0 | lower=-10, upper=10, effort=100, velocity=1000 |
| joint3 / revolute | link2 → link3 | xyz=-0.264 0 0, rpy=3.1416 0 0 | xyz=0 1 0 | lower=-10, upper=10, effort=100, velocity=1000 |
| joint4 / revolute | link3 → link4 | xyz=0.245 0 -0.056, rpy=0 0 0 | xyz=0 1 0 | lower=-10, upper=10, effort=100, velocity=1000 |
| joint5 / revolute | link4 → link5 | xyz=0.06775 0.0005 -0.0865, rpy=0 0 0 | xyz=0 0 1 | lower=-10, upper=10, effort=100, velocity=1000 |
| joint6 / revolute | link5 → link6 | xyz=0.02895 0 0.0865, rpy=-3.1416 0 0 | xyz=1 0 0 | lower=-3.14, upper=3.14, effort=100, velocity=1000 |
| hand_to_camera_mount / fixed | link6 → camera_base | xyz=0.057 0 0, rpy=0 0 -3.1416 | not specified in the source configuration | not specified in the source configuration |
| camera_joint / fixed | camera_base → camera | xyz=-0.0275 0.0 0.05, rpy=0 0.3491 3.141592653589793 | not specified in the source configuration | not specified in the source configuration |
| joint7 / prismatic | link6 → link7 | xyz=0.08657 0.024896 -0.0002436, rpy=0 0 0 | xyz=0 1 0 | lower=0, upper=0.044, effort=100, velocity=1000 |
| joint8 / prismatic | link6 → link8 | xyz=0.08657 -0.0249 -0.00024366, rpy=0 0 0 | xyz=0 -1 0 | lower=0, upper=0.044, effort=100, velocity=1000 |
