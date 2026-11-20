"""Tests for road motion, channel rates, and shared OFDMA scheduling."""

import unittest

from src.vec import (
    Destination,
    DeterministicChannelModel,
    DispatchDecision,
    PolylineRoute,
    QueueAwareOFDMA,
    RadioConfig,
    StepAction,
    SyntheticRoadMobility,
    VecEnvConfig,
    VecQueueEnv,
    VehicleRouteConfig,
)
from src.vec.queues import QueueStage
from src.vec.radio import LinkRequest


class MobilityTests(unittest.TestCase):
    def test_vehicle_follows_polyline_and_reflects_at_endpoint(self):
        route = PolylineRoute(
            ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0))
        )
        mobility = SyntheticRoadMobility(
            1, routes=(VehicleRouteConfig(route, speed_mps=15.0),)
        )

        self.assertEqual(mobility.advance(1.0)[0], (10.0, 5.0, 0.0))
        self.assertEqual(mobility.advance(1.0)[0], (10.0, 0.0, 0.0))

    def test_loop_route_includes_return_road_segment(self):
        route = PolylineRoute(
            ((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0)),
            loop=True,
        )
        mobility = SyntheticRoadMobility(
            1, routes=(VehicleRouteConfig(route, speed_mps=25.0),)
        )
        x, y, _ = mobility.advance(1.0)[0]
        self.assertAlmostEqual(x, 6.464466094, places=6)
        self.assertAlmostEqual(y, 6.464466094, places=6)


class RadioTests(unittest.TestCase):
    def test_rate_decreases_with_distance(self):
        channel = DeterministicChannelModel()
        rsu = (0.0, 0.0, 0.0)
        near_gain = channel.terrestrial_gain((5.0, 0.0, 0.0), rsu)
        far_gain = channel.terrestrial_gain((100.0, 0.0, 0.0), rsu)

        self.assertGreater(near_gain, far_gain)
        self.assertGreater(channel.rate(near_gain, 1.0e6), channel.rate(far_gain, 1.0e6))

    def test_scheduler_is_deterministic_queue_weighted_and_orthogonal(self):
        scheduler = QueueAwareOFDMA()
        requests = (
            LinkRequest(3, 0, (0.0, 0.0, 0.0), 100.0, 100.0, 0.0, 10.0),
            LinkRequest(1, 1, (0.0, 0.0, 0.0), 300.0, 300.0, 0.0, 10.0),
        )
        allocations = scheduler.schedule_rsu(
            requests, (0.0, 0.0, 0.0), total_bandwidth_hz=40.0, slot_duration=1.0
        )

        self.assertEqual(tuple(item.task_id for item in allocations), (1, 3))
        self.assertAlmostEqual(sum(item.bandwidth_hz for item in allocations), 40.0)
        by_task = {item.task_id: item for item in allocations}
        self.assertAlmostEqual(by_task[1].bandwidth_hz, 30.0)
        self.assertAlmostEqual(by_task[3].bandwidth_hz, 10.0)

    def test_contact_loss_retains_upload_and_received_staging(self):
        route = PolylineRoute(((0.0, 0.0, 0.0), (20.0, 0.0, 0.0)))
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(0.0,),
            uav_cpu_hz=(),
            rsu_uplink_bps=(0.0,),
            uav_uplink_bps=(),
            radio=RadioConfig(rsu_coverage_m=5.0),
            rsu_bandwidth_hz=(1.0,),
            uav_bandwidth_hz=(),
            vehicle_routes=(VehicleRouteConfig(route, speed_mps=10.0),),
            rsu_positions=((0.0, 0.0, 0.0),),
        )
        env = VecQueueEnv(config)
        task_id = env.submit_task(0, input_bits=1000.0, cpu_cycles=1000.0, deadline=100)
        first = env.step(
            StepAction(
                dispatches=(DispatchDecision(0, task_id, Destination.rsu(0)),)
            )
        )
        upload_after_contact = env.get_queue(
            QueueStage.UPLOAD_RSU, 0
        ).amount_for(task_id)
        staging_after_contact = env.get_queue(
            QueueStage.RECEIVE_RSU, 0
        ).amount_for(task_id)

        self.assertEqual(first.vehicle_positions_start[0], (0.0, 0.0, 0.0))
        self.assertEqual(first.vehicle_positions_end[0], (10.0, 0.0, 0.0))
        self.assertEqual(len(first.radio_allocations), 1)
        self.assertGreater(upload_after_contact, 0.0)
        self.assertGreater(staging_after_contact, 0.0)

        second = env.step()
        self.assertEqual(second.radio_allocations, ())
        self.assertAlmostEqual(
            env.get_queue(QueueStage.UPLOAD_RSU, 0).amount_for(task_id),
            upload_after_contact,
        )
        self.assertAlmostEqual(
            env.get_queue(QueueStage.RECEIVE_RSU, 0).amount_for(task_id),
            staging_after_contact,
        )
        self.assertAlmostEqual(env.total_backlog_cycles, 1000.0)

    def test_dispatch_to_out_of_coverage_node_is_rejected(self):
        config = VecEnvConfig(
            vehicle_cpu_hz=(0.0,),
            rsu_cpu_hz=(0.0,),
            uav_cpu_hz=(),
            rsu_uplink_bps=(0.0,),
            uav_uplink_bps=(),
            radio=RadioConfig(rsu_coverage_m=5.0),
            rsu_bandwidth_hz=(1.0,),
            uav_bandwidth_hz=(),
            vehicle_static_positions=((10.0, 0.0, 0.0),),
            rsu_positions=((0.0, 0.0, 0.0),),
        )
        env = VecQueueEnv(config)
        task_id = env.submit_task(0, 10.0, 10.0, 10.0)
        with self.assertRaisesRegex(ValueError, "outside current coverage"):
            env.step(
                StepAction(
                    dispatches=(DispatchDecision(0, task_id, Destination.rsu(0)),)
                )
            )


if __name__ == "__main__":
    unittest.main()
