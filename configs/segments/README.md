# Segment files

One YAML file per circuit, created by `km segments` from the recorded reference laps.
Positions are shares of the lap (0 to 1) on the track spline of the simulator.

The proposed boundaries have to be checked against a track map before they are used:
corners that belong together (a chicane, for example) can be merged by hand, and the
names can be replaced by the usual corner names. A file that exists is never overwritten
unless `km segments --force` is used.
