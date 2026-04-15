from abc import ABC, abstractmethod
from typing import List
import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from pfvtr.msg import FloatList
from pfvtr.msg import SensorsOutput, ImageList, Features
from pfvtr.srv import SetDist, Representations

NAVIGATION_QOS = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE
)


"""
These are base classes for sensor modules in this package
"""

class DisplacementEstimator(ABC):
    """
    Base class for displacement estimator
    Extend this to add new estimator, the main method which must be implemented is "_displacement_message_callback"
    """

    def __init__(self, logger=None):
        self.supported_message_type = None
        self._log = logger
        if not self.health_check():
            self._log.error("Displacement estimator health check was not successful")
            raise Exception("Displacement Estimator health check failed")

    def displacement_message_callback(self, msg: object) -> List[np.ndarray]:
        if not isinstance(msg, self.supported_message_type) or self.supported_message_type is None:
            self._log.error(f"Incorrect type of message in displacement estimator {type(msg)} vs {self.supported_message_type}")
            raise Exception("Wrong message type")
        return self._displacement_message_callback(msg)

    @abstractmethod
    def _displacement_message_callback(self, msg: object) -> List[np.ndarray]:
        """
           returns list of histograms (displacement probabilities) -> there could be one histogram or multiple
           """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError


class RelativeDistanceEstimator(ABC):
    """
     Abtract method for estimating the relative distance traveled from last measurement
     Extend this to add new estimator, the main method which must be implemented is "_rel_dist_message_callback"
     """

    def __init__(self, logger=None):
        self.supported_message_type = None
        self._log = logger
        if not self.health_check():
            self._log.error("Relative distance estimator health check was not successful")
            raise Exception("Rel Dist Estimator health check failed")

    def rel_dist_message_callback(self, msg: object) -> float:
        if not isinstance(msg, self.supported_message_type) or self.supported_message_type is None:
            self._log.error(f"Incorrect type of message in relative distance estimator {type(msg)} vs {self.supported_message_type}")
            raise Exception("Wrong message type")
        return self._rel_dist_message_callback(msg)

    @abstractmethod
    def _rel_dist_message_callback(self, msg: object) -> float:
        """
           returns float value which tells by how much the robot moved
           """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError



# TODO: absolute distance is probably subclass of probability distance, rework this into one class
class AbsoluteDistanceEstimator(ABC):

    """
    Abstract method for estimating the absolute distance traveled - needs also handle header because of synchronization
    Extend this to add new estimator, the main method which must be implemented is "_abs_dist_message_callback"
    """

    def __init__(self, logger=None):
        self.supported_message_type = None
        self._distance = None
        self.header = None
        self._log = logger
        if not self.health_check():
            self._log.error("Absolute distance estimator health check was not successful")
            raise Exception("Abs Dist Estimator health check failed")

    def abs_dist_message_callback(self, msg: object) -> float:
        if self._distance is None:
            self._log.error("If you want to use absolute distance sensor - you have to set the distance first!")
            raise Exception("The distance must be set first")
        if not isinstance(msg, self.supported_message_type) or self.supported_message_type is None:
            self._log.error(f"Incorrect type of message in absolute distance estimator {type(msg)} vs {self.supported_message_type}")
            raise Exception("Wrong message type")
        return self._abs_dist_message_callback(msg)

    def set_dist(self, dist):
        self._log.info(f"Setting distance to: {dist}")
        self._distance = self._set_dist(dist)

    def _set_dist(self, dist):
        return dist

    @abstractmethod
    def _abs_dist_message_callback(self, msg: object) -> float:
        """
        increment the absolute distance self._distance in here
        returns floats -> distance absolute value
        """
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError


class ProbabilityDistanceEstimator(ABC):
    """
    Abstract method for estimating the absolute distance traveled.
    Extend this to add new estimator, the main method which must be implemented is "_abs_dist_message_callback"
    """

    def __init__(self, logger=None):
        self._distance = None
        self.supported_message_type = None
        self._log = logger
        if self.health_check():
            self._log.error("Absolute distance estimator health check was not successful")
            raise Exception("Abs Dist Estimator health check failed")

    def prob_dist_message_callback(self, msg: object) -> List[float]:
        if self._distance is None:
            self._log.error("If you want to use absolute distance sensor - you have to set the distance first!")
            raise Exception("The distance must be set first")
        if not isinstance(msg, self.supported_message_type) or self.supported_message_type is None:
            self._log.error(f"Incorrect type of message in probabilistic distance estimator {type(msg)} vs {self.supported_message_type}")
            raise Exception("Wrong message type")
        return self._prob_dist_message_callback(msg)

    def set_dist(self, dist):
        self._distance = dist

    @abstractmethod
    def _prob_dist_message_callback(self, msg: object) -> List[float]:
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        """
        returns list of floats -> probability of traveled distance
        """
        raise NotImplementedError



class RepresentationsCreator(ABC):

    def __init__(self):
        self.supported_message_type = None
        self.health_check()

    def to_feature(self, inputs: Representations.Request) -> Representations.Response:
        resp = Representations.Response()
        resp.features = self._to_feature(inputs.images)
        return resp

    def from_feature(self, feature: Features) -> object:
        return self._from_feature(feature)

    @abstractmethod
    def _to_feature(self, inputs: object) -> Features:
        raise NotImplementedError

    @abstractmethod
    def _from_feature(self, feature: Features):
        raise NotImplementedError

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError




class SensorFusion(ABC):
    """
    Abstract method for the sensor fusion!
    """

    def __init__(
        self,
        node: Node,
        type_prefix: str = "repeat",
        abs_dist_est: AbsoluteDistanceEstimator = None,
        rel_dist_est: RelativeDistanceEstimator = None,
        prob_dist_est: ProbabilityDistanceEstimator = None,
        rel_align_est: DisplacementEstimator = None,
        abs_align_est: DisplacementEstimator = None,
        repr_creator: RepresentationsCreator = None,
    ):
        if type_prefix not in ["teach", "repeat"]:
            node.get_logger().error("Fusion method must be created for teach or repeat phase")
            raise Exception("Not properly initialized fusion method")

        self.node = node
        self._log = node.get_logger()
        self.type_prefix = type_prefix

        self.output_dist = node.create_publisher(
            SensorsOutput, f"{type_prefix}/output_dist", NAVIGATION_QOS
        )
        self.output_align = node.create_publisher(
            SensorsOutput, f"{type_prefix}/output_align", NAVIGATION_QOS
        )

        self.set_distance_srv = node.create_service(
            SetDist, f"{type_prefix}/set_dist", self.set_distance
        )
        self.set_alignment_srv = node.create_service(
            SetDist, f"{type_prefix}/set_align", self.set_alignment
        )

        self.header = None
        self.distance = None
        self.alignment = None
        self.distance_std = None
        self.alignment_std = None
        self.map = 0

        self.abs_dist_est = abs_dist_est
        self.rel_dist_est = rel_dist_est
        self.prob_dist_est = prob_dist_est
        self.abs_align_est = abs_align_est
        self.rel_align_est = rel_align_est
        self.repr_creator = repr_creator



    def publish_dist(self):
        """
             publish distance as a float in meters - we need always header for time synchronization!
        """
        out = SensorsOutput()
        if self.header is not None:
            out.header = self.header
        if self.distance is not None:
            out.output = float(self.distance)
            out.output_uncertainty = float(self.distance_std)
        else:
            out.output = 0.0
            out.output_uncertainty = -1.0
        out.map = self.map
        self.output_dist.publish(out)

    def publish_align(self):
        """
        publish alignment as a float from -1.0 to 1.0
        """
        out = SensorsOutput()
        if self.header is not None:
            out.header = self.header
        if self.alignment is not None:
            out.output = float(self.alignment)
            out.output_uncertainty = float(self.alignment_std)
        else:
            out.output = 0.0
            out.output_uncertainty = -1.0
        out.map = self.map
        self.output_align.publish(out)



    def set_distance(self, request: SetDist.Request, response: SetDist.Response):
        self.distance = request.dist
        self.distance_std = 0.0
        self.map = getattr(request, 'map_num', 0)
        if self.abs_dist_est is not None:
            self.abs_dist_est.set_dist(self.distance)
        if self.prob_dist_est is not None:
            self.prob_dist_est.set_dist(self.distance)
        self.publish_dist()
        return response

    def set_alignment(self, request: SetDist.Request, response: SetDist.Response):
        self.alignment = request.dist
        self.alignment_std = 0.0
        self.publish_align()
        return response


    # callback for services
    def create_representations(self, msg):
        return self.repr_creator.to_feature(msg)

    def process_rel_alignment(self, request, response):
        result = self._process_rel_alignment(request)
        if result is not None:
            response.histograms = []
            for hist in result.histograms:
                msg = FloatList()
                msg.data = [float(x) for x in hist]
                response.histograms.append(msg)
        return response

    def process_abs_alignment(self, msg):
        if self.alignment is not None:
            self._process_abs_alignment(msg)
            self.publish_align()

    def process_rel_distance(self, msg):
        if self.distance is not None:
            self._process_rel_distance(msg)
            self.publish_dist()

    def process_abs_distance(self, msg):
        if self.distance is not None:
            self._process_abs_distance(msg)
            self.publish_dist()

    def process_prob_distance(self, msg):
        if self.distance is not None:
            self._process_prob_distance(msg)
            self.publish_dist()

    @abstractmethod
    def _process_rel_alignment(self, request, response):
        raise NotImplementedError

    @abstractmethod
    def _process_abs_alignment(self, msg):
        raise NotImplementedError

    @abstractmethod
    def _process_rel_distance(self, msg):
        raise NotImplementedError

    @abstractmethod
    def _process_abs_distance(self, msg):
        raise NotImplementedError

    @abstractmethod
    def _process_prob_distance(self, msg):
        raise NotImplementedError