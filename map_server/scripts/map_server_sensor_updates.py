#!/usr/bin/env python3

import rospy
import std_msgs.msg
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from nav_msgs.msg import OccupancyGrid,MapMetaData
from geometry_msgs.msg import Pose,Point,Quaternion
from tf.transformations import quaternion_from_euler
import tf2_ros
from jsk_recognition_msgs.msg import BoundingBox,BoundingBoxArray

import numpy as np
import os
import cv2
import yaml

class MapServerSensorUpdates:

    def __init__(self) -> None:
        
        #specify ROS params - for loading/publishing the map
        self.map_path = rospy.get_param(param_name='~map_path')
        self.map_topic_static = rospy.get_param(
            param_name='~map_topic_static',
            default='/map/static')
        self.map_topic_dynamic = rospy.get_param(
            param_name='~map_topic_dynamic',
            default='/map/dynamic')
        self.map_frame_id = rospy.get_param(
            param_name='~map_frame_id',
            default='map')
        self.update_rate_Hz = float(
                rospy.get_param(param_name='~update_rate_Hz',
                                default=20)
            )
        
        #specify ROS params - for using PointCloud2 message to update the map
        self.point_cloud_updates_enabled = rospy.get_param(
            param_name='~point_cloud_updates_enabled',
            default=False
        )
        self.point_cloud_updates_topic = rospy.get_param(
            param_name="~point_cloud_updates_topic",
            default='/lidar_cloudout'
        )

        #static map characteristics
        self.map_resolution:float = 0.0
        self.map_width:int = 0
        self.map_height:int = 0
        self.map_origin:Pose = Pose()
        self.static_map:np.ndarray = None
        self.dynamic_map:np.ndarray = None
        self.map_metadata:MapMetaData = MapMetaData()
        self.map_x_indicies = None
        self.map_y_indicies = None
        
        #map publisher
        self.map_pub_static:rospy.Publisher = None
        self.map_pub_dynamic:rospy.Publisher = None

        #tf transformers and listeners
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        #point cloud listener
        self.point_cloud_sub:rospy.Subscriber = None
        self.point_cloud_mask_latest = None


        #initialize the map
        self.init_map_server()

        #start the map update functions if enabled
        if self.point_cloud_updates_enabled:
            self.point_cloud_sub_init()

        #start a time to run the node at the given intervals
        rospy.Timer(
            period=rospy.Duration(1/self.update_rate_Hz),
            callback=self.update_map
        )
        return
    
    ####################################################################
    # Initialization Functions
    ####################################################################
    
    def init_map_server(self):

        #load the map
        self.load_map_from_yaml()
        self.init_map_metadata()
        rospy.loginfo("Map loaded with shape: [{},{}]".format(
            self.map_height,
            self.map_width
        ))
        #initialize the publishers
        self.map_pubs_init()

        #publish the static map
        self.map_pub_static.publish(
            self.get_occupancy_grid_msg(
                map_data=self.static_map
            )
        )
    
    def map_pubs_init(self):

        self.map_pub_static = rospy.Publisher(
            name=self.map_topic_static,
            data_class=OccupancyGrid,
            queue_size=1,
            latch=True
        )
        rospy.loginfo(
            "Static map server publishing at topic: {}".format(
                self.map_topic_static
            ))


        self.map_pub_dynamic = rospy.Publisher(
            name=self.map_topic_dynamic,
            data_class=OccupancyGrid,
            queue_size=1,
            latch=True
        )
        rospy.loginfo(
            "Dynamic map server publishing at topic: {}".format(
                self.map_topic_dynamic
            ))

        return
    
    def load_map_from_yaml(self):

        #check to make sure that the map path is valid
        assert os.path.exists(self.map_path), "Couldn't find map at: {}".format(self.map_path)

        with open(self.map_path,'r') as file:
            config = yaml.safe_load(file)
        
        #get the map file name
        map_name = config["image"]
        # map_dir = os.path.dirname(self.map_path)
        # map_pgm_path = os.path.join(map_dir,map_name)

        #get the key map terms
        self.map_resolution = config["resolution"]
        self.map_origin = self.get_map_origin_pose(config["origin"])

        #read the image
        img = cv2.imread(map_name,cv2.IMREAD_GRAYSCALE).astype('uint8')
        
        #determine map dimmensions
        self.map_height,self.map_width = img.shape

        #initialize the static map
        occupied_regions = (img == 0)
        free_regions = (img>250)
        unknown_regions = ~(occupied_regions | free_regions)

        self.static_map = np.zeros(shape=(self.map_height,self.map_width),dtype=np.int8)
        self.static_map[occupied_regions] = 100
        self.static_map[free_regions] = 0
        self.static_map[unknown_regions] = -1

        self.static_map = np.flipud(self.static_map)

        #save the indicies
        self.map_x_indicies = \
            np.arange(self.map_width) * self.map_resolution + config["origin"][0]
        
        self.map_y_indicies = \
            np.arange(self.map_height) * self.map_resolution + config["origin"][1]

        return
    
    def init_map_metadata(self):

        #initialize the map metadata
        self.map_metadata = MapMetaData()
        self.map_metadata.map_load_time = rospy.Time.now()
        self.map_metadata.resolution = self.map_resolution
        self.map_metadata.width = self.map_width
        self.map_metadata.height = self.map_height
        self.map_metadata.origin = self.map_origin

        return
    
    def get_map_origin_pose(self,origin:np.ndarray)->Pose:

        origin_pose = Pose()

        #set position
        origin_pose.position.x = origin[0]
        origin_pose.position.y = origin[1]
        origin_pose.position.z = 0.0

        #set the quaternion
        quat_orientation = quaternion_from_euler(
            ai = 0.0,
            aj = 0.0,
            ak = origin[2])
        
        origin_pose.orientation.x = quat_orientation[0]
        origin_pose.orientation.y = quat_orientation[1]
        origin_pose.orientation.z = quat_orientation[2]
        origin_pose.orientation.w = quat_orientation[3]

        return origin_pose


    ####################################################################
    # Map Updates
    ####################################################################

    def update_map(self,event):
        
        #start with the static map
        self.dynamic_map = self.static_map

        if self.point_cloud_updates_enabled and \
            (self.point_cloud_mask_latest is not None):
            
            #denote the occupied regions
            self.dynamic_map[self.point_cloud_mask_latest] = 100

        #convert map to map message
        occupancy_grid_msg = self.get_occupancy_grid_msg(
            map_data=self.dynamic_map
        )

        #publish the occupancy grid map
        self.map_pub_dynamic.publish(occupancy_grid_msg)
        rospy.loginfo("Dynamic Map Updated")
        return
    
    def get_occupancy_grid_msg(self,map_data:np.ndarray):

        occupancy_grid = OccupancyGrid()

        #define the header
        occupancy_grid.header.stamp = rospy.Time.now()
        occupancy_grid.header.frame_id = self.map_frame_id

        #define the metadata
        occupancy_grid.info = self.map_metadata
        occupancy_grid.info.map_load_time = rospy.Time.now()

        #import the data
        occupancy_grid.data = map_data.flatten(order='C')

        return occupancy_grid
    
    ####################################################################
    # Updating map from the point cloud
    ####################################################################
    def point_cloud_sub_init(self):

        self.point_cloud_sub = rospy.Subscriber(
            name=self.point_cloud_updates_topic,
            data_class=PointCloud2,
            callback=self.point_cloud_sub_callback,
            queue_size=1
        )

        return
    
    def point_cloud_sub_callback(self,msg:PointCloud2):

        #transform the point cloud to the map frame
        point_cloud_transformed = self.point_cloud_transform_to_map_frame(msg)

        if point_cloud_transformed is not None:

            self.point_cloud_mask_latest = \
                self.point_cloud_get_occupied_mask(point_cloud_transformed)

        return

    def point_cloud_transform_to_map_frame(self,point_cloud:PointCloud2)->PointCloud2:

        #try to get the transformation
        try:

            tf_lidar_to_map = self.tf_buffer.lookup_transform(
                target_frame=self.map_frame_id,
                source_frame=point_cloud.header.frame_id,
                time=rospy.Time(secs=0), #get the latest transformation
                timeout=rospy.Duration(secs=1.0)
            )
        
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn("Error looking up transformation: {}".format(ex))
            return None
        try:

            transformed_cloud = do_transform_cloud(
                cloud=point_cloud,
                transform=tf_lidar_to_map
            )

            return transformed_cloud
        
        except tf2_ros.TransformException as ex:

            rospy.logwarn("Error transforming point cloud: {}".format(ex))

            return None
    
    def point_cloud_get_occupied_mask(self,point_cloud_msg:PointCloud2):

        msg_points = pc2.read_points(point_cloud_msg,skip_nans=True)

        points = np.array(list(msg_points))

        #get only the 2d points
        points = points[:,:2]

        x_indicies = np.floor(
            (points[:,0] - self.map_origin.position.x) / self.map_resolution
            ).astype(int)

        y_indicies = (
            (points[:,1] - self.map_origin.position.y) / self.map_resolution
            ).astype(int)
        
        #make sure that the indicies are valid
        valid_x_idxs = (x_indicies >= 0) & (x_indicies <= self.map_width)
        valid_y_idxs = (y_indicies >= 0) & (y_indicies <= self.map_height)
        valid_idxs = valid_x_idxs & valid_y_idxs

        #filter the points to only valid points
        x_indicies = x_indicies[valid_idxs]
        y_indicies = y_indicies[valid_idxs]
        
        mask = np.zeros(
            shape=(self.map_height,self.map_width),
            dtype=bool
        )

        mask[y_indicies,x_indicies] = True

        return mask

    


def main():

    rospy.init_node("MapServerSensorUpdates",anonymous=True)

    try:
        map_server_sensor_updates = MapServerSensorUpdates()
        rospy.spin()
    except rospy.ROSException:
        pass

if __name__=='__main__':
    main()