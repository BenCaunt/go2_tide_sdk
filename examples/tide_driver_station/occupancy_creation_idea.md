- Occupancy grid in go2_sensors_node
- Should manifest itself in the existing tide node.  Needs to be efficient 
- There should be green tiles for free space and red for occupied. 
- It should show the robot position and orientation in the rerun visualization 
- The data should be easily indexible for path planning later (path planning impl not included in this work)
- Should use a filter such as ensuring that in a given cell there are at least n lidar points with an obstacle height signature.  Make this default bound points which are greater than 0.2 meters off the ground and less than 1 meter high.  Set the cutoff at 5 for now.
- The data coming from the unitree lidar is interesting.  there is already some preprocessing to a degree and I believe it is already in the global frame and conistent! the problem is there is a sunset as the robot moves away from a scanned area and it decays but a simple cache could probably fix this problem.  It should be updated when the robot starts returning new data in that location though. 


If you plan on importing other files you make use this so tide knows how to find them:

from tide.core.utils import add_project_root_to_path

add_project_root_to_path(__file__)
