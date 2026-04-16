#!/usr/bin/env python3
import numpy as np
from scipy import interpolate
import torch as t

import rclpy
from rclpy.time import Time

from base_classes import DisplacementEstimator, RelativeDistanceEstimator, AbsoluteDistanceEstimator, \
    SensorFusion, ProbabilityDistanceEstimator, RepresentationsCreator

# from backends.nn_policy.model import PolicyNet

from pfvtr.srv import Alignment, SetDist
from pfvtr.msg import FloatList, SensorsInput, ImageList


"""
Here should be placed all classes for fusion of sensor processing
"""


def _builtin_stamp_to_rclpy_time(stamp, node_clock) -> Time:
    """
    Convert ROS2 builtin_interfaces/msg/Time to rclpy.time.Time.
    """
    # stamp.sec, stamp.nanosec
    return Time(seconds=int(stamp.sec), nanoseconds=int(stamp.nanosec), clock_type=node_clock.clock_type)


def _duration_to_sec(duration) -> float:
    """
    Convert rclpy.duration.Duration to seconds (float).
    """
    return float(duration.nanoseconds) / 1e9


class BearnavClassic(SensorFusion):
    def __init__(
        self,
        node,
        type_prefix: str,
        abs_align_est: DisplacementEstimator,
        abs_dist_est: AbsoluteDistanceEstimator,
        repr_creator: RepresentationsCreator,
        rel_align_est: DisplacementEstimator,
    ):
        super().__init__(
            node,
            type_prefix,
            abs_align_est=abs_align_est,
            abs_dist_est=abs_dist_est,
            rel_align_est=rel_align_est,
            repr_creator=repr_creator,
        )
        self._log = node.get_logger()

    def _process_rel_alignment(self, request: Alignment.Request):
        histogram = self.rel_align_est.displacement_message_callback(request.input)
        out = Alignment.Response()
        out.histograms = histogram
        return out

    def _process_abs_alignment(self, msg):
        # if msg.map_features[0].shape[0] > 1:
        #     rospy.logwarn("Bearnav classic can process only one image")
        if len(msg.map_histograms) == 0:
            self._log.warn("No map_histograms received!")
            return
        histogram = np.array(msg.map_histograms[0].values).reshape(msg.map_histograms[0].shape)
        self._log.info(f"Histogram shape: {histogram.shape}, max: {np.max(histogram):.4f}, argmax: {np.argmax(histogram)}")
        self.alignment = (np.argmax(histogram) - np.size(histogram) // 2) / (np.size(histogram) // 2)
        self._log.info("Current displacement: " + str(self.alignment))
        # self.publish_align()

    def _process_rel_distance(self, msg):
        self._log.error("This function is not available for this fusion class")
        raise Exception("Bearnav Classic does not support relative distance")

    def _process_abs_distance(self, msg):
        self.distance = self.abs_dist_est.abs_dist_message_callback(msg)

        self.header = self.abs_dist_est.header
        # self.publish_dist()

    def _process_prob_distance(self, msg):
        self._log.error("This function is not available for this fusion class")
        raise Exception("Bearnav Classic does not support probability of distances")


class VisualOnly(SensorFusion):
    def __init__(
        self,
        node,
        type_prefix: str,
        abs_align_est: DisplacementEstimator,
        prob_dist_est: ProbabilityDistanceEstimator,
        repr_creator: RepresentationsCreator,
    ):
        super().__init__(
            node,
            type_prefix,
            abs_align_est=abs_align_est,
            prob_dist_est=prob_dist_est,
            repr_creator=repr_creator,
        )
        self._log = node.get_logger()

    def _process_rel_alignment(self, msg):
        self._log.warn("This function is not available for this fusion class")
        raise Exception("Visual only does not support relative alignment")

    def _process_abs_alignment(self, msg: SensorsInput):
        hists = np.array(msg.live_histograms[0].values).reshape(msg.live_histograms[0].shape)
        hist = np.max(hists, axis=0)
        half_size = np.size(hist) / 2.0
        self.alignment = float(np.argmax(hist) - (np.size(hist) // 2.0)) / half_size
        # self.publish_align()

    def _process_rel_distance(self, msg):
        self._log.warn("This function is not available for this fusion class")
        raise Exception("Visual only does not support relative distance")

    def _process_abs_distance(self, msg):
        self._log.warn("This function is not available for this fusion class")
        raise Exception("Visual only does not support absolute distance")

    def _process_prob_distance(self, msg):
        # TODO: this method usually publishes with too low frequency to control the spot
        dists = msg.map_distances
        probs = self.prob_dist_est.prob_dist_message_callback(msg)
        # TODO: add some interpolation to more cleanly choose between actions - more fancy :)
        self.distance = max(dists[np.argmax(probs)], 0.05)
        self._log.info("Predicted dist: " + str(self.distance) + " and alignment: " + str(self.alignment))
        # self.publish_dist()

class PF2D(SensorFusion):
    # First dimension - traveled distance
    # Second dimension - alignment
    # Third dimension - maps

    def __init__(
        self,
        node,
        type_prefix: str,
        particles_num: int,
        odom_error: float,
        odom_init_std: float,
        align_beta: float,
        align_init_std: float,
        particles_frac: int,
        choice_beta: float,
        add_random: float,
        debug: bool,
        abs_align_est: DisplacementEstimator,
        rel_align_est: DisplacementEstimator,
        rel_dist_est: RelativeDistanceEstimator,
        repr_creator: RepresentationsCreator,
    ):
        super().__init__(
            node,
            type_prefix,
            abs_align_est=abs_align_est,
            rel_align_est=rel_align_est,
            rel_dist_est=rel_dist_est,
            repr_creator=repr_creator,
        )
        self._node = node
        self._log = node.get_logger()
        self._clock = node.get_clock()


        self.zero_dim = False
        self.one_dim = False
        self.use_map_trans = False
        self.fallback_bearnav = False
        self.init_distance = 0.0
        self.rng = np.random.default_rng()

        self.odom_error = odom_error
        self.odom_init_std = odom_init_std
        self.align_init_std = align_init_std

        self.particles_num = particles_num
        self.particles_frac = particles_frac
        self.add_rand = add_random
        self.last_image = None
        self.last_odom = None
        self.particles = None
        self.last_time = None
        self.traveled_dist = 0.0
        self.particle_prob = None
        self.coords = None
        self.last_hists = None
        self.map = 0
        self.last_map_transition_time = self._clock.now()
        self.map_num = 1

        self._min_align_noise = 0.01
        self._map_trans_time = 5.0

        self.BETA_align = align_beta
        self.BETA_choice = choice_beta

        # For debugging
        self.debug = debug
        if debug:
            self.particles_pub = node.create_publisher(FloatList, "particles", 1)


    def set_distance(self, request: SetDist.Request, response: SetDist.Response):
        self.fallback_bearnav = False
        response = super(PF2D, self).set_distance(request, response)

        var = (self.odom_init_std, self.align_init_std, 0)
        dst = self.distance
        self.init_distance = self.distance
        self.particles = np.transpose(
            np.ones((3, self.particles_num)).transpose() * np.array((dst, 0, 0))
            + self.rng.normal(loc=(0, 0, 0), scale=var, size=(self.particles_num, 3)))
        # self.particles = self.particles - np.mean(self.particles, axis=-1, keepdims=True)
        self.map_num = request.map_num
        self.particles[2] = np.random.randint(low=0, high=self.map_num, size=(self.particles_num,))
        self.particle_prob = np.ones(self.particles_num) / self.particles_num
        self.last_image = None
        self._get_coords()

        self._log.info(
            "Particles reinitialized at position " + str(self.distance) + "m"
            + " with alignment " + str(self.alignment)
        )
        return response

    def _process_rel_alignment(self, request: Alignment.Request):
        histogram = self.rel_align_est.displacement_message_callback(request.input)
        out = Alignment.Response()
        out.histograms = histogram
        return out

    def _create_trans_matrix(self, array, map_num):
        transitions = np.zeros((map_num, map_num))
        curr_idx = 0
        for i in range(map_num):
            for j in range(i, map_num):
                transitions[i, j] = array[curr_idx]
                transitions[j, i] = array[curr_idx]
                curr_idx += 1
        for i in range(map_num):
            transitions[i] = self._numpy_softmax(transitions[i], self.BETA_choice)
        self._log.warn(str(transitions))
        return transitions

    def _get_time_diff(self, timestamps: list):
        out = []
        for i in range(len(timestamps) - 1):
            t0 = _builtin_stamp_to_rclpy_time(timestamps[i], self._clock)
            t1 = _builtin_stamp_to_rclpy_time(timestamps[i + 1], self._clock)
            out.append(_duration_to_sec(t1 - t0))
        return out

    def _process_abs_alignment(self, msg):
        # Parse all data from the incoming message
        curr_time = _builtin_stamp_to_rclpy_time(msg.header.stamp, self._clock)
        self.header = msg.header
        if self.last_time is None:
            self.last_time = curr_time
            return

        dists = np.array(msg.map_distances)
        len_per_map = np.size(dists) // self.map_num

        if len_per_map == 1:
            self._log.error("!!!Only one image matched - fallback to bearnav classic!!!")
            self.fallback_bearnav = True
            # self.distance = self.init_distance

        if self.fallback_bearnav:
            histogram = np.array(msg.map_histograms[0].values).reshape(msg.map_histograms[0].shape)
            self.alignment = (np.argmax(histogram) - np.size(histogram) // 2) / (np.size(histogram) // 2)
            self._log.info("Fallback displacement: " + str(self.alignment))
            return

        hists = np.array(msg.map_histograms[0].values).reshape(msg.map_histograms[0].shape)
        self.last_hists = hists
        map_trans = np.array(msg.map_transitions[0].values).reshape(msg.map_transitions[0].shape)
        live_hist = np.array(msg.live_histograms[0].values).reshape(msg.live_histograms[0].shape)
        hist_width = hists.shape[-1]
        shifts = np.round(np.array(msg.map_offset) * (hist_width // 2)).astype(int)
        hists = np.roll(hists, shifts, -1) # not sure if last dim should be rolled like this
        curr_img_diff = self._sample_hist([live_hist])
        curr_time_diff = _duration_to_sec(curr_time - self.last_time)
        timestamps = msg.map_timestamps
        traveled = self.traveled_dist

        # Divide incoming data according the map affiliation
        trans_per_map = len_per_map - 1
        if len(dists) % msg.map_num > 0:
            # TODO: this assumes that there is same number of features comming from all the maps (this does not have to hold when 2*map_len < lookaround)
            # however the mapmaker was updated so that the new maps should have always the same number of images, unless some major error occurs
            self._log.warn("!!!!!!!!!!!!!!!!!! One map has more images than other !!!!!!!!!!!!!!!!")
            return

        map_trans = [map_trans[trans_per_map * map_idx:trans_per_map * (map_idx + 1)] for map_idx in range(self.map_num)]
        hists = [hists[len_per_map * map_idx:len_per_map * (map_idx + 1)] for map_idx in range(self.map_num)]
        dists = np.array([dists[len_per_map * map_idx:len_per_map * (map_idx + 1)] for map_idx in range(self.map_num)])
        timestamps = [timestamps[len_per_map * map_idx:len_per_map * (map_idx + 1)] for map_idx in range(self.map_num)]
        time_diffs = [self._get_time_diff(timestamps[map_idx]) for map_idx in range(self.map_num)]
        if self.map_num > 1:
            # transition matrix for between maps
            map_matrix = msg.map_transitions

        # if len(hists) < 2 or len(trans) != len(hists) - 1 or len(dists) != len(hists) or len(trans) == 0:
        #     rospy.logwarn("Invalid input sizes for particle filter!")
        #     return

        if abs(traveled) < 0.001:
            # this is when odometry is slower than the estimator
            self.last_time = curr_time
            # rospy.logwarn(
            #     "Not enough movement detected for particle filter update!\n" + "traveled: " + str(traveled) + "," + str(
            #         np.var(curr_img_diff)))
            return

        if self.particles.shape[-1] > self.particles_num:
            self._log.warn("errorneous state - too much particles")
            self.particles = self.particles[:, :self.particles_num]

        # motion step
        for map_idx in range(self.map_num):
            # get map transition for each particle
            map_particle_mask = self.particles[2, :] == map_idx
            # centers of distances
            mat_dists = np.transpose(np.matrix((dists[map_idx, 1:] + dists[map_idx, :-1]) / 2.0))
            p_distances = np.matrix(self.particles[0, map_particle_mask])
            particles_in_map = np.sum(map_particle_mask)
            # rospy.logwarn(np.argmin(np.abs(mat_dists - p_distances)))
            closest_transition = np.transpose(np.argmin(np.abs(mat_dists - p_distances), axis=0))
            traveled_fracs = float(curr_time_diff) / np.array(time_diffs[map_idx])
            # rospy.loginfo("traveled fracs:" + str(traveled_fracs))
            # Monte carlo sampling of transitions
            trans = -self._sample_hist(map_trans[map_idx])
            trans_per_particle = trans[closest_transition.squeeze(), np.arange(particles_in_map)].transpose()
            frac_per_particle = traveled_fracs[closest_transition]
            # generate new particles
            out = []
            # rolls = np.random.rand(self.particles.shape[1])
            # indices = self._first_nonzero(np.matrix(trans_cumsum_per_particle) >= np.transpose(np.matrix(rolls)), 1)
            trans_diff = np.array(trans_per_particle * frac_per_particle)
            align_shift = curr_img_diff.transpose()[map_particle_mask] + trans_diff

            # debugging
            # self._log.warn(str(closest_transition))
            # self._log.warn("map_trans " + str(np.argmax(map_trans[0], axis=-1)))
            # self._log.warn("live: " + str(np.mean(curr_img_diff)))
            # self._log.warn("map: " + str(np.mean(trans_diff)))
            # self._log.warn("curr_time: " + str(curr_time_diff))
            # self._log.warn("map_time " + str(time_diffs[map_idx]))

            # distance is not shifted because it is shifted already in odometry step
            particle_shifts = np.concatenate((np.zeros(trans_diff.shape), align_shift), axis=1)

            moved_particles = np.transpose(self.particles[:2, map_particle_mask]) + particle_shifts + \
                              self.rng.normal(
                                  loc=(0, 0),
                                  scale=(self.odom_error * traveled, 0),
                                  size=(particles_in_map, 2)
                              )
            out.append(moved_particles)
            self.particles[:2, map_particle_mask] = moved_particles.transpose()

            # TODO: this has to be updated for joint map state
            if self.use_map_trans and _duration_to_sec(self._clock.now() - self.last_map_transition_time) > self._map_trans_time \
                    and hists is not None:
                random_indices = self.rng.choice(np.arange(self.particles_num), int(self.particles_num // 5))
                random_particles = self.particles[2, random_indices]
                # move particles in map space, but not too often - time limit
                self.last_map_transition_time = self._clock.now()
                for map_id in range(msg.maps[1]):
                    random_particles[random_particles == map_id] = self.rng.choice(
                        np.arange(msg.maps[1]),
                        int(np.sum(random_particles == map_id)),
                        p=map_matrix[map_id]
                    )
                self.particles[2, random_indices] = random_particles
                # rospy.logwarn("Motion step finished!")

        # add randomly spawned particles
        if self.add_rand > 0:
            new = []
            tmp = np.zeros((3, int(self.particles_num * self.add_rand)))
            tmp[0, :] = self.rng.uniform(low=np.mean(dists[:, 0]), high=np.mean(dists[:, -1]),
                                         size=(1, int(self.particles_num * self.add_rand)))
            tmp[1, :] = self.rng.uniform(low=-0.5, high=0.5, size=(1, int(self.particles_num * self.add_rand)))
            tmp[2, :] = np.random.randint(low=0, high=self.map_num, size=(1, int(self.particles_num * self.add_rand)))
            new.append(tmp.transpose())
            new.append(self.particles.transpose())
            self.particles = np.concatenate(new).transpose()
            if self.particle_prob is not None:
                # set the lowest possible probability to all added particles
                self.particle_prob = np.concatenate(
                    [self.particle_prob, np.zeros((self.particles[0].size - self.particles_num), )]
                )

        # sensor step
        particle_prob = np.zeros(self.particles.shape[-1])
        self.particles[1] = np.clip(self.particles[1], -1.0, 1.0)  # more than 0% overlap is nonsense
        for map_idx in range(self.map_num):
            map_particle_mask = self.particles[2] == map_idx
            map_masked_particles = self.particles[:, map_particle_mask]

            interp_f = interpolate.RectBivariateSpline(
                dists[map_idx], np.linspace(-1.0, 1.0, hist_width), hists[map_idx], kx=1
            )
            particle_prob[map_particle_mask] = interp_f(map_masked_particles[0], map_masked_particles[1], grid=False)

        self.particle_prob = particle_prob
        self.particle_prob[self.particle_prob < 0] = 0.0   # lower than 0.0 probability - should not happen though

        # perform some normalization and resample the particles via roulette wheel
        # particle_prob -= particle_prob.min()
        # particle_prob /= particle_prob.sum()
        softmaxed_probs = self._numpy_softmax(self.particle_prob, self.BETA_choice)
        chosen_indices = self.rng.choice(np.shape(self.particles)[1], int(self.particles_num), p=softmaxed_probs)
        # self._log.warn((self.particles[2, chosen_indices])
        self.particle_prob = self.particle_prob[chosen_indices]
        self.particles = self.particles[:, chosen_indices]

        # publish filtering output ------------------------------------------------------------------
        self.last_image = msg.live_features
        self.last_time = curr_time
        self.traveled_dist = 0.0
        self._get_coords()  # this updates the values which are published continuously

        #  self._log.warn(np.array((dist_diff, hist_diff)))
        # visualization & debugging
        if self.debug:
            #for
            particles_out = self.particles.flatten()
            particles_out = np.concatenate([particles_out, self.coords.flatten()])
            m = FloatList()
            m.data = list(particles_out)
            self.particles_pub.publish(m)
            # rospy.loginfo("Outputted position: " + str(np.mean(self.particles[0, :])) + " +- " + str(np.std(self.particles[0, :])))
            # rospy.loginfo("Outputted alignment: " + str(np.mean(self.particles[1, :])) + " +- " + str(np.std(self.particles[1, :])) + " with transitions: " + str(np.mean(curr_img_diff))
            #               + " and " + str(np.mean(trans_diff)))

            # rospy.logwarn(
            #     "Finished processing - everything took: " + str((rospy.Time.now() - msg.header.stamp).to_sec()) + " secs")

    def _process_rel_distance(self, msg):
        # only increment the distance
        dist = self.rel_dist_est.rel_dist_message_callback(msg)
        if dist is not None and dist >= 0.005:
            if self.fallback_bearnav:
                self.distance += dist
            else:
                self.particles[0] += dist
                self._get_coords()
                self.traveled_dist += dist

    def _process_abs_distance(self, msg):
        self._log.warn("This function is not available for this fusion class")
        raise Exception("PF2D does not support absolute distance")

    def _process_prob_distance(self, msg):
        self._log.warn("This function is not available for this fusion class")
        raise Exception("PF2D does not support distance probabilities")

    def _numpy_softmax(self, arr, beta):
        tmp = np.exp(arr * beta) / np.sum(np.exp(arr * beta))
        return tmp

    def _first_nonzero(self, arr, axis, invalid_val=-1):
        mask = arr != 0
        return np.array(np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val))

    def _get_coords(self):
        # TODO: implement better estimate - histogram voting!
        # coords = np.mean(self.particles, axis=1)
        tmp_particles = self.particles
        if self.particle_prob is not None:
            self.coords = self._get_weighted_mean_pos(np.copy(self.particles), np.copy(self.particle_prob))
            # self.coords = self._histogram_voting()
            # self.coords = self.particles[:2, np.argmax(self.particle_prob)]
            if self.one_dim:
                # for testing not using 2nd dim
                self.coords[1] = (np.argmax(np.max(self.last_hists, axis=0)) - self.last_hists[0].size // 2) / \
                                 self.last_hists[0].size
            if np.size(self.particle_prob) == np.size(self.particles[2]): # avoid inconsistencies
                maps = []
                for i in range(self.map_num):
                    maps.append(np.sum(self.particle_prob[self.particles[2] == i]))
                ind = np.argmax(maps)
                # rospy.logwarn(maps)
                self.map = int(ind)
        else:
            self.coords = [0.0, 0.0]

        if self.coords[0] < 0.0:
            # the estimated distance cannot really be less than 0.0 - fixing for action repeating
            self._log.warn("Mean of particles is less than 0.0 - moving them forwards!")
            self.particles[0, :] -= self.coords[0] - 0.01  # add one centimeter for numeric issues
        stds = np.std(tmp_particles, axis=1)
        self.distance = self.coords[0]
        self.alignment = self.coords[1]
        self.distance_std = stds[0]
        self.alignment_std = stds[1]

    def _diff_from_hist(self, hist):
        half_size = np.size(hist) / 2.0
        curr_img_diff = ((np.argmax(hist) - (np.size(hist) // 2.0)) / half_size)
        return curr_img_diff

    def _sample_hist(self, hists: list):
        """
        sample from histogram per each particle ...
        """
        hist_size = hists[0].shape[-1]
        return np.array([
            self.rng.choice(
                np.linspace(start=-1, stop=1, num=hist_size),
                int(self.particles_num),
                p=self._numpy_softmax(hists[idx], self.BETA_align),
            )
            for idx, curr_trans_sample in enumerate(hists)
        ])

    def _sample_maxs(self, hists: list):
        return np.array([
            np.ones(self.particles_num) * self._diff_from_hist(curr_trans_sample)
            for idx, curr_trans_sample in enumerate(hists)
        ])

    def _get_mean_pos(self):
        return np.mean(self.particles, axis=1)

    def _get_weighted_mean_pos(self, particles, particle_prob):
        # TODO: this method can yield an error when class variables are changed in process - make copies
        align_span = 0.5   # crop of particles to estimate alignment
        predictive = 0.0   # make alignment slightly predictive
        if np.size(particles[0]) != np.size(particle_prob):
            particles = particles[:, :self.particles_num]
            particle_prob = particle_prob[:self.particles_num]
            self._log.warn("Ommiting particles spawned this round - concurrency issue")

        dist = np.sum(particles[0] * particle_prob) / np.sum(particle_prob)
        mask = (particles[0] < (dist + align_span + predictive)) * (particles[0] > (dist - align_span + predictive))
        p_num = np.sum(mask)
        if p_num < 50:
            self._log.warn("Only " + str(p_num) + " particles used for alignment estimate - could be very noisy")
        if np.isnan(particle_prob).any() or np.isnan(particles).any() or np.sum(particle_prob[mask]) == 0.0:
            self._log.warn("Somehow NaN in particles or probs!")
            self._log.warn(str(particle_prob[mask]))
            self._log.warn(str(particles))
            self._log.warn(str(np.sum(particle_prob[mask])))
            return np.array((dist, self.alignment))

        align = np.sum(particles[1, mask] * particle_prob[mask]) / np.sum(particle_prob[mask])
        return np.array((dist, align))
        # weighted_particles = particles[:2] * np.tile(particle_prob, (2, 1))
        # out = np.sum(weighted_particles, axis=1) / np.sum(particle_prob)
        # return out

    def _histogram_voting(self):
        hist, x, y = np.histogram2d(self.particles[0], self.particles[1], bins=20, weights=self.particle_prob)
        indices = np.unravel_index(np.argmax(hist, axis=None), hist.shape)
        return np.array([x[indices[0]], y[indices[1]]])

    def _get_median_pos(self):
        return np.median(self.particles, axis=1)


# class NNPolicy(SensorFusion):
#     def __init__(
#         self,
#         node,
#         type_prefix: str,
#         min_control_dist: float,
#         abs_align_est: DisplacementEstimator,
#         rel_align_est: DisplacementEstimator,
#         rel_dist_est: RelativeDistanceEstimator,
#         repr_creator: RepresentationsCreator,
#     ):
#         super().__init__(
#             node,
#             type_prefix,
#             abs_align_est=abs_align_est,
#             rel_align_est=rel_align_est,
#             rel_dist_est=rel_dist_est,
#             repr_creator=repr_creator,
#         )
#         self._node = node
#         self._log = node.get_logger()
#         self._clock = node.get_clock()
# 
#         self.device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
#         self.net = PolicyNet(device=self.device)
#         self.min_control_dist = 0.1
#         self.last_time = None
#         self.map_num = 1
#         self.input_size = 5
#         self.dist_span = 8
# 
#     def _process_rel_alignment(self, request: Alignment.Request):
#         histogram = self.rel_align_est.displacement_message_callback(request.input)
#         out = Alignment.Response()
#         out.histograms = histogram
#         return out
# 
#     def _process_abs_alignment(self, msg):
#         curr_time = _builtin_stamp_to_rclpy_time(msg.header.stamp, self._clock)
#         self.header = msg.header
#         if self.last_time is None:
#             self.last_time = curr_time
#             return
# 
#         hists = np.array(msg.map_histograms[0].values).reshape(msg.map_histograms[0].shape)
#         self.last_hists = hists
#         map_trans = np.array(msg.map_transitions[0].values).reshape(msg.map_transitions[0].shape)
#         live_hist = np.array(msg.live_histograms[0].values).reshape(msg.live_histograms[0].shape)
#         hist_width = hists.shape[-1]
#         shifts = np.round(np.array(msg.map_offset) * (hist_width // 2)).astype(int)
#         hists = np.roll(hists, shifts, -1)   # not sure if last dim should be rolled like this
#         dists = np.array(msg.map_distances)
#         timestamps = msg.map_timestamps
# 
#         # Divide incoming data according the map affiliation
#         len_per_map = np.size(dists) // self.map_num
#         trans_per_map = len_per_map - 1
#         if len(dists) % msg.map_num > 0:
#             # TODO: this assumes that there is same number of features comming from all the maps (this does not have to hold when 2*map_len < lookaround)
#             self._log.warn("!!!!!!!!!!!!!!!!!! One map has more images than other !!!!!!!!!!!!!!!!")
#             return
# 
#         map_trans = [map_trans[trans_per_map * map_idx:trans_per_map * (map_idx + 1)] for map_idx in range(self.map_num)]
#         hists = [hists[len_per_map * map_idx:len_per_map * (map_idx + 1)] for map_idx in range(self.map_num)]
#         dists = np.array([dists[len_per_map * map_idx:len_per_map * (map_idx + 1)] for map_idx in range(self.map_num)])
#         timestamps = [timestamps[len_per_map * map_idx:len_per_map * (map_idx + 1)] for map_idx in range(self.map_num)]
# 
#         in_diff = self.input_size - hists[0].shape[0]
#         if in_diff > 0:
#             in_append = np.zeros((in_diff, 1024))
#             if dists[0][0] < 2.0:
#                 out_hists = np.concatenate((in_append, hists[0]), axis=0)
#                 out_map_trans = np.concatenate((in_append, map_trans[0]), axis=0)
#             else:
#                 out_hists = np.concatenate((hists[0], in_append), axis=0)
#                 out_map_trans = np.concatenate((map_trans[0], in_append), axis=0)
#         else:
#             out_hists = hists[0]
#             out_map_trans = map_trans[0]
# 
#         # self.data = (dists, out_hists, out_map_trans, np.expand_dims(live_hist, axis=0))
#         data = (dists, out_hists, out_map_trans)
#         img_data = self.parse_hists(data[1:])
#         img_pos = self.process_distance(data[0])
# 
#         obs = t.cat([img_data, img_pos]).float()
#         action = self.net.get_action(obs)
#         self._log.warn("NN output: " + str(action["action"]))
#         self.distance -= action["action"][0, 1].cpu().detach().numpy()
#         self.alignment = action["action"][0, 0].cpu().detach().numpy()
#         self.last_time = curr_time
# 
#     def process_distance(self, img_dists):
#         center = int((self.dist_span * 10) / 2.0)
# 
#         # GET IMG POS
#         imgs_pos = t.zeros(self.dist_span * 10 + 1, device=self.device)
#         img_dist_idxs = np.round((self.distance - img_dists) * 10)
#         for index_diff in img_dist_idxs[0]:
#             if abs(index_diff) < center:
#                 imgs_pos[int(index_diff) + center] = 0.2
#         return imgs_pos
# 
#     def parse_hists(self, obs):
#         cat_tesnors = []
#         # max_map_img = t.argmax(t.max(t.tensor(obs[0]), dim=1)[0])
#         # rospy.logwarn("MAX MAP: " + str(max_map_img))
#         for n_obs in obs:
#             flat_tensor = t.tensor(n_obs[:, 256:-256], device=self.device).flatten()
#             flat_tensor = self.resize_histogram(flat_tensor)
#             norm_flat_tesnsor = (flat_tensor - t.mean(flat_tensor)) / t.std(flat_tensor)
#             cat_tesnors.append(norm_flat_tesnsor.squeeze(0))
#         new_obs = t.cat(cat_tesnors, dim=0).float()
#         return new_obs
# 
#     def resize_histogram(self, x):
#         # RESIZE BY FACTOR of 8
#         reshaped_x = x.view(1, -1, 8)
#         # Sum along the second dimension to aggregate every 8 bins into one
#         resized_x = reshaped_x.sum(dim=2)
#         return resized_x
# 
#     def _process_rel_distance(self, msg):
#         # only increment the distance
#         dist = self.rel_dist_est.rel_dist_message_callback(msg)
#         self.distance += dist
# 
#     def _process_abs_distance(self, msg):
#         self._log.warn("This function is not available for this fusion class")
#         raise Exception("NNPolicy does not support absolute distance")
# 
#     def _process_prob_distance(self, msg):
#         self._log.error("This function is not available for this fusion class")
#         raise Exception("NNPolicy does not support probability of distances")
