# Solver, timing and sensors

Physics dt is 0.004 s; decimation is 1; render interval is 10 physics steps.
Control and public observation collection are 25 Hz (0.04 s); policy horizon is
300 control steps. Native task action limit is 500; the policy loop terminates
earlier according to its fixed termination order. Gravity is [0,0,-9.81] m/s².
External forces apply every position iteration. Contact processing is enabled
after scene construction; cube contact-report threshold is zero. Contact/scoring
observation does not advance physics. Cube motion velocity is measured from poses
at physics substeps; solver velocities are separate host records.

## PhysX scene settings

Solver type is 1 (TGS). Position iteration minimum/maximum are 1/255;
velocity iteration minimum/maximum are 0/255. CCD, stabilization and enhanced
determinism are false. Bounce threshold velocity is 0.5 m/s; friction offset
threshold is 0.04 m; friction correlation distance is 0.025 m.

GPU capacity declarations: max rigid contacts 8388608, max rigid patches 163840,
found/lost pairs 2097152, found/lost aggregate pairs 33554432, total aggregate pairs
2097152, collision stack 100000000 bytes, heap 67108864 bytes, temporary buffer
16777216 bytes, max partitions 8, max soft-body contacts 1048576, max particle
contacts 1048576. These declarations describe the configured source; they do not
imply that a CPU adapter launches GPU simulation.

## Sensors

Proprioception provides arm joint positions/velocities, normalized gripper opening,
and world end-effector pose; physical encoder noise is not specified in the source
configuration. Contact reports provide cube–table, cube–plate, cube–robot and
per-finger contacts for source truth and scoring. Public tactile reports per-pad
normal compression from the unfiltered net force of all collision partners:
max(0,-F dot n), with inward world-frame normal n. Tactile is separate from scoring
contacts. Sensor spatial extent, contact-force noise and bandwidth beyond control
sampling are not specified in the source configuration; sensor plumbing arrives
with the batch runtime.

Head camera: Gemini_345Lg pinhole, position [0,-0.41,1.308], orientation Euler
[30,0,0] degrees, resolution 640x480, focal length 10.0 mm, horizontal/vertical
aperture 22.212/14.266 mm, clipping [0.005,10.0] m. Each wrist has a D435 pinhole
camera mounted on link camera, local position [0,0,0.001], Euler [0,-80,-90]
degrees, resolution 640x480, focal length 13.0 mm, aperture 20.955/15.71625 mm,
clipping [0.0001,10.0] m. RGB capture is enabled on head and both wrists on CPU;
depth, approximate depth and intrinsic/extrinsic matrix collection are disabled.

The deployable actor receives one reset-image cube/plate XY detection and fixed
calibration for rendered runs. Non-rendered source training/evaluation use simulated
reset detection with independent Gaussian errors clipped at three calibrated
standard deviations. Calibration is a host-supplied frozen artifact, not a physical
sensor model specified in the source configuration. The actor otherwise uses clock,
commanded pad position and opening; privileged object state is critic-only.

Reset-image detection uses cube/plate pixel ROIs (x0,y0,x1,y1) of
(380,215,466,286) and (250,175,371,281), requiring at least 40 and 400 pixels.
Red hue is within 12 degrees of red with saturation >=0.65 and value >=0.45;
white saturation <=0.15 and value >=0.75. Per-object homographies map image
centroids to world XY. Calibration sets each noise standard deviation to the
larger of 1.5 times the largest-axis validation RMS and a floor of 0.002 m for
the cube or 0.003 m for the plate. Nominal Z attachments for actor detections
are table top plus cube half-side (0.7775 m) and plate centre (0.7828 m).
