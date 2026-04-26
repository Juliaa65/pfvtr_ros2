#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import os
import glob

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from pfvtr.action import MapMaker, MapRepeater
from pfvtr.msg import FeaturesList


class VTRControlGUI(Node):
    def __init__(self):
        super().__init__('vtr_control_gui')

        self.declare_parameter("navigation_method", "classic")
        self.navigation_method = self.get_parameter("navigation_method").value
        if self.navigation_method not in ("classic", "pf2d"):
            self.get_logger().warn(
                f"Invalid navigation_method '{self.navigation_method}' "
                "- falling back to 'classic'"
            )
            self.navigation_method = "classic"

        self.mapmaker_client = ActionClient(self, MapMaker, '/pfvtr/mapmaker')
        self.repeater_client = ActionClient(self, MapRepeater, '/pfvtr/repeater')
        self.controller_param_client = self.create_client(
            SetParameters, '/pfvtr/controller/set_parameters'
        )

        self.mapping_goal_handle = None
        self.repeating_goal_handle = None
        self.last_mapping_goal_was_start = False

        self.root = tk.Tk()
        self.root.title("VT&R Control Panel")
        self.root.geometry("500x720")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        self.actions_tab = ttk.Frame(self.notebook)
        self.control_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.actions_tab, text="Actions")
        self.notebook.add(self.control_tab, text="Control")

        self.setup_mapping_frame(self.actions_tab)
        self.setup_repeating_frame(self.actions_tab)
        self.setup_status_bar(self.actions_tab)
        self.setup_hz_frame(self.actions_tab)
        self.setup_control_frame(self.control_tab)

        self.refresh_maps_list()

        # Periodic ROS2 spinning
        self.root.after(20, self.spin_ros)
    
    def setup_mapping_frame(self, parent):
        mapping_frame = ttk.LabelFrame(parent, text="Mapping Controls", padding=10)
        mapping_frame.pack(fill='x', padx=10, pady=5)
        
        # Map name
        ttk.Label(mapping_frame, text="Map Name:").grid(row=0, column=0, sticky='w', pady=2)
        self.map_name_var = tk.StringVar(value="my_first_map")
        ttk.Entry(mapping_frame, textvariable=self.map_name_var, width=30).grid(row=0, column=1, pady=2)
        
        # Save images checkbox
        self.save_imgs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(mapping_frame, text="Save Images for Visualization",
                       variable=self.save_imgs_var).grid(row=1, column=0, columnspan=2, sticky='w', pady=2)

        # Map step
        ttk.Label(mapping_frame, text="Map Step (m):").grid(row=2, column=0, sticky='w', pady=2)
        self.map_step_var = tk.StringVar(value="1.0")
        ttk.Entry(mapping_frame, textvariable=self.map_step_var, width=10).grid(row=2, column=1, sticky='w', pady=2)

        # Source map
        ttk.Label(mapping_frame, text="Source Map:").grid(row=3, column=0, sticky='w', pady=2)
        self.source_map_var = tk.StringVar(value="")
        ttk.Entry(mapping_frame, textvariable=self.source_map_var, width=30).grid(row=3, column=1, pady=2)

        # Record backward: uses the rear camera configured via the
        # `camera_back_topic` launch argument; the mapmaker post-processes
        # the finished map so it can be repeated forward with the front
        # camera after a 180° turn. Goal is rejected if no rear camera is
        # configured.
        self.record_backward_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mapping_frame, text="Record Backward (rear camera)",
                       variable=self.record_backward_var).grid(row=4, column=0, columnspan=2, sticky='w', pady=2)

        # Buttons
        button_frame = ttk.Frame(mapping_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=10)
        
        self.start_mapping_btn = ttk.Button(button_frame, text="START MAPPING", 
                                            command=lambda: self.send_mapping_goal(start=True))
        self.start_mapping_btn.pack(side='left', padx=5)
        
        self.stop_mapping_btn = ttk.Button(button_frame, text="STOP MAPPING", 
                                           command=lambda: self.send_mapping_goal(start=False), 
                                           state='disabled')
        self.stop_mapping_btn.pack(side='left', padx=5)
    
    def setup_repeating_frame(self, parent):
        repeating_frame = ttk.LabelFrame(parent, text="Repeating Controls", padding=10)
        repeating_frame.pack(fill='x', padx=10, pady=5)
        
        # Map name with dropdown
        ttk.Label(repeating_frame, text="Map Name:").grid(row=0, column=0, sticky='w', pady=2)
        
        map_select_frame = ttk.Frame(repeating_frame)
        map_select_frame.grid(row=0, column=1, sticky='w', pady=2)
        
        self.repeat_map_name_var = tk.StringVar()
        self.map_combobox = ttk.Combobox(map_select_frame, textvariable=self.repeat_map_name_var, 
                                         width=22, state='readonly')
        self.map_combobox.pack(side='left')
        
        refresh_btn = ttk.Button(map_select_frame, text="Refresh", width=8,
                                command=self.refresh_maps_list)
        refresh_btn.pack(side='left', padx=5)
        
        # Start position
        ttk.Label(repeating_frame, text="Start Position:").grid(row=1, column=0, sticky='w', pady=2)
        self.start_pos_var = tk.StringVar(value="0.0")
        ttk.Entry(repeating_frame, textvariable=self.start_pos_var, width=15).grid(row=1, column=1, sticky='w', pady=2)
        
        # End position
        ttk.Label(repeating_frame, text="End Position:").grid(row=2, column=0, sticky='w', pady=2)
        self.end_pos_var = tk.StringVar(value="0.0")
        ttk.Entry(repeating_frame, textvariable=self.end_pos_var, width=15).grid(row=2, column=1, sticky='w', pady=2)
        
        # Traversals
        ttk.Label(repeating_frame, text="Traversals:").grid(row=3, column=0, sticky='w', pady=2)
        self.traversals_var = tk.StringVar(value="1")
        ttk.Entry(repeating_frame, textvariable=self.traversals_var, width=5).grid(row=3, column=1, sticky='w', pady=2)
        
        # Checkboxes
        self.null_cmd_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(repeating_frame, text="Null Commands", 
                       variable=self.null_cmd_var).grid(row=4, column=0, columnspan=2, sticky='w', pady=2)
        
        self.use_dist_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(repeating_frame, text="Use Distance-Based Navigation", 
                       variable=self.use_dist_var).grid(row=5, column=0, columnspan=2, sticky='w', pady=2)
        
        # Image publish mode — constraint depends on the navigation_method
        # launch parameter: classic requires 0, pf2d requires >= 1.
        if self.navigation_method == "classic":
            image_pub_label = "Image Publish Mode (classic: 0):"
            image_pub_default = "0"
            image_pub_state = "disabled"
        else:  # pf2d
            image_pub_label = "Image Publish Mode (pf2d: ≥1):"
            image_pub_default = "1"
            image_pub_state = "normal"
        ttk.Label(repeating_frame, text=image_pub_label).grid(row=6, column=0, sticky='w', pady=2)
        self.image_pub_var = tk.StringVar(value=image_pub_default)
        ttk.Entry(repeating_frame, textvariable=self.image_pub_var, width=5,
                  state=image_pub_state).grid(row=6, column=1, sticky='w', pady=2)
        
        # Send button
        self.send_repeat_btn = ttk.Button(repeating_frame, text="SEND REPEAT COMMAND", 
                                          command=self.send_repeating_goal)
        self.send_repeat_btn.grid(row=7, column=0, columnspan=2, pady=10)
    
    def setup_status_bar(self, parent):
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.status_text = tk.Text(status_frame, height=10, width=50, state='disabled')
        self.status_text.pack(fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(status_frame, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)

    def setup_hz_frame(self, parent):
        hz_frame = ttk.LabelFrame(parent, text="Topic Rates", padding=5)
        hz_frame.pack(fill='x', padx=10, pady=5)

        self.hz_label = ttk.Label(hz_frame, text="Cam: --   LiveRepr: --   Odom: --")
        self.hz_label.pack(side='left', padx=5)

        self.hz_btn = ttk.Button(hz_frame, text="Measure", command=self.start_hz_measurement)
        self.hz_btn.pack(side='right', padx=5)

        self.odom_count = 0
        self.repr_count = 0
        self._odom_sub = None
        self._repr_sub = None

    def setup_control_frame(self, parent):
        controller_frame = ttk.LabelFrame(parent, text="Controller Parameters", padding=10)
        controller_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(controller_frame, text="turn_gain:").grid(row=0, column=0, sticky='w', pady=4)
        self.turn_gain_var = tk.StringVar(value="1.0")
        ttk.Entry(controller_frame, textvariable=self.turn_gain_var, width=10).grid(
            row=0, column=1, sticky='w', pady=4, padx=5
        )
        ttk.Button(
            controller_frame, text="Apply",
            command=lambda: self._apply_controller_param("turn_gain", self.turn_gain_var.get()),
        ).grid(row=0, column=2, sticky='w', pady=4)

        ttk.Label(controller_frame, text="velocity_gain:").grid(row=1, column=0, sticky='w', pady=4)
        self.velocity_gain_var = tk.StringVar(value="1.0")
        ttk.Entry(controller_frame, textvariable=self.velocity_gain_var, width=10).grid(
            row=1, column=1, sticky='w', pady=4, padx=5
        )
        ttk.Button(
            controller_frame, text="Apply",
            command=lambda: self._apply_controller_param("velocity_gain", self.velocity_gain_var.get()),
        ).grid(row=1, column=2, sticky='w', pady=4)

        ttk.Label(
            controller_frame,
            text="Sets parameters live on /pfvtr/controller via SetParameters service.",
            foreground="gray",
        ).grid(row=2, column=0, columnspan=3, sticky='w', pady=(8, 0))

        self.control_status_var = tk.StringVar(value="")
        ttk.Label(controller_frame, textvariable=self.control_status_var,
                  foreground="blue").grid(row=3, column=0, columnspan=3, sticky='w', pady=(4, 0))

    def _apply_controller_param(self, name, raw_value):
        try:
            value = float(raw_value)
        except ValueError:
            msg = f"ERROR: '{name}' must be numeric, got '{raw_value}'"
            self.log_status(msg)
            self.control_status_var.set(msg)
            return

        if not self.controller_param_client.wait_for_service(timeout_sec=1.0):
            msg = "ERROR: /pfvtr/controller/set_parameters not available (controller node running?)"
            self.log_status(msg)
            self.control_status_var.set(msg)
            return

        req = SetParameters.Request()
        param = Parameter()
        param.name = name
        param.value = ParameterValue()
        param.value.type = ParameterType.PARAMETER_DOUBLE
        param.value.double_value = value
        req.parameters = [param]

        future = self.controller_param_client.call_async(req)
        future.add_done_callback(
            lambda fut, n=name, v=value: self._param_set_done(fut, n, v)
        )

    def _param_set_done(self, future, name, value):
        try:
            response = future.result()
        except Exception as exc:
            msg = f"ERROR setting {name}: {exc}"
            self.log_status(msg)
            self.control_status_var.set(msg)
            return

        if not response.results:
            msg = f"ERROR setting {name}: empty result"
            self.log_status(msg)
            self.control_status_var.set(msg)
            return
        result = response.results[0]
        if result.successful:
            msg = f"Set {name} = {value}"
            self.log_status(msg)
            self.control_status_var.set(msg)
        else:
            msg = f"REJECTED {name}={value}: {result.reason}"
            self.log_status(msg)
            self.control_status_var.set(msg)

    def start_hz_measurement(self):
        self.hz_btn['state'] = 'disabled'
        self.hz_label.config(text="Measuring...")
        self.odom_count = 0
        self.repr_count = 0

        hz_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self._odom_sub = self.create_subscription(Odometry, '/odometry_publisher', self._odom_cb, hz_qos)
        self._repr_sub = self.create_subscription(FeaturesList, '/pfvtr/live_representation', self._repr_cb, hz_qos)
        self.root.after(3000, self.finish_hz_measurement)

    def _odom_cb(self, msg):
        self.odom_count += 1

    def _repr_cb(self, msg):
        self.repr_count += 1

    def finish_hz_measurement(self):
        if self._odom_sub is not None:
            self.destroy_subscription(self._odom_sub)
            self._odom_sub = None
        if self._repr_sub is not None:
            self.destroy_subscription(self._repr_sub)
            self._repr_sub = None

        odom_hz = self.odom_count / 3.0
        repr_hz = self.repr_count / 3.0
        cam_hz = odom_hz

        self.hz_label.config(text=f"Cam: {cam_hz:.1f} Hz   LiveRepr: {repr_hz:.1f} Hz   Odom: {odom_hz:.1f} Hz")
        self.hz_btn['state'] = 'normal'
    
    def log_status(self, message):
        self.status_text.configure(state='normal')
        self.status_text.insert('end', message + '\n')
        self.status_text.see('end')
        self.status_text.configure(state='disabled')
    
    def refresh_maps_list(self):
        maps = []
        for item in os.listdir('.'):
            if os.path.isdir(item) and os.path.exists(os.path.join(item, 'params')):
                maps.append(item)
        
        maps.sort()
        self.map_combobox['values'] = maps
        if maps:
            self.map_combobox.current(0)
        self.log_status(f"Found {len(maps)} maps: {', '.join(maps) if maps else 'None'}")
    
    def send_mapping_goal(self, start):
        self.log_status(f"DEBUG: send_mapping_goal called with start={start}")
        
        if not self.mapmaker_client.wait_for_server(timeout_sec=2.0):
            self.log_status("ERROR: MapMaker action server not available!")
            return
        
        goal = MapMaker.Goal()
        goal.save_imgs_for_viz = self.save_imgs_var.get()
        goal.map_name = self.map_name_var.get()
        goal.start = start
        try:
            goal.map_step = float(self.map_step_var.get())
        except ValueError:
            goal.map_step = 1.0
        goal.source_map = self.source_map_var.get()
        goal.record_backward = self.record_backward_var.get()
        
        action_text = "START" if start else "STOP"
        self.log_status(f"Sending {action_text} MAPPING goal...")
        
        # Track what type of goal we're sending
        self.last_mapping_goal_was_start = start
        
        # Update button states based on action
        if start:
            # Starting mapping: disable START, enable STOP
            self.log_status("Starting mapping - disabling START button, enabling STOP button")
            self.start_mapping_btn['state'] = 'disabled'
            self.stop_mapping_btn['state'] = 'normal'
            self.root.update_idletasks()
        else:
            # Stopping mapping: disable STOP, enable START
            self.log_status("Stopping mapping - enabling START button, disabling STOP button")
            self.start_mapping_btn['state'] = 'normal'
            self.stop_mapping_btn['state'] = 'disabled'
            self.root.update_idletasks()
        
        future = self.mapmaker_client.send_goal_async(
            goal,
            feedback_callback=self.mapping_feedback_callback
        )
        future.add_done_callback(self.mapping_goal_response_callback)
    
    def mapping_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.log_status("Mapping goal REJECTED")
            # Reset buttons on rejection
            self.start_mapping_btn['state'] = 'normal'
            self.stop_mapping_btn['state'] = 'disabled'
            self.root.update_idletasks()
            return
        
        self.log_status("Mapping goal ACCEPTED")
        self.mapping_goal_handle = goal_handle
        
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.mapping_result_callback)
    
    def mapping_feedback_callback(self, feedback_msg):
        pass
    
    def mapping_result_callback(self, future):
        result = future.result().result
        
        # MapMaker returns success immediately after starting
        # We only reset buttons when STOP goal completes, or if any goal fails
        if self.last_mapping_goal_was_start:
            # START goal completed
            if result.success:
                # Mapping started successfully - keep STOP button enabled!
                self.log_status("Mapping started successfully - click STOP MAPPING to end")
                # Do NOT reset button states here!
            else:
                # Failed to start mapping - reset to initial state
                self.log_status("ERROR: Failed to start mapping")
                self.start_mapping_btn['state'] = 'normal'
                self.stop_mapping_btn['state'] = 'disabled'
                self.root.update_idletasks()
        else:
            # STOP goal completed - mapping has stopped
            self.log_status(f"Mapping stopped - Success: {result.success}")
            # Reset buttons to initial state
            self.start_mapping_btn['state'] = 'normal'
            self.stop_mapping_btn['state'] = 'disabled'
            self.root.update_idletasks()
        
        self.mapping_goal_handle = None
    
    def send_repeating_goal(self):
        if not self.repeater_client.wait_for_server(timeout_sec=2.0):
            self.log_status("ERROR: Repeater action server not available!")
            return
        
        if not self.repeat_map_name_var.get():
            messagebox.showwarning("Warning", "Please select a map from the list!")
            return
        
        goal = MapRepeater.Goal()
        try:
            goal.start_pos = float(self.start_pos_var.get())
            goal.end_pos = float(self.end_pos_var.get())
            goal.traversals = int(self.traversals_var.get())
            goal.image_pub = int(self.image_pub_var.get())
        except ValueError:
            self.log_status("ERROR: Invalid numeric values in fields!")
            return

        if self.navigation_method == "classic" and goal.image_pub != 0:
            self.log_status("ERROR: classic mode requires image_pub == 0")
            return
        if self.navigation_method == "pf2d" and goal.image_pub < 1:
            self.log_status("ERROR: pf2d mode requires image_pub >= 1")
            return

        goal.map_name = self.repeat_map_name_var.get()
        goal.null_cmd = self.null_cmd_var.get()
        goal.use_dist = self.use_dist_var.get()
        
        self.log_status(f"Sending REPEAT goal for map: {goal.map_name}")
        
        future = self.repeater_client.send_goal_async(
            goal,
            feedback_callback=self.repeating_feedback_callback
        )
        future.add_done_callback(self.repeating_goal_response_callback)
        
        self.send_repeat_btn.configure(state='disabled')
    
    def repeating_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.log_status("Repeating goal REJECTED")
            self.send_repeat_btn.configure(state='normal')
            return
        
        self.log_status("Repeating goal ACCEPTED - Robot is now navigating...")
        self.repeating_goal_handle = goal_handle
        
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.repeating_result_callback)
    
    def repeating_feedback_callback(self, feedback_msg):
        pass
    
    def repeating_result_callback(self, future):
        result = future.result().result
        self.log_status(f"Repeating completed - Success: {result.success}")
        self.send_repeat_btn.configure(state='normal')
        self.repeating_goal_handle = None
    
    def spin_ros(self):
        # Drain pending ROS work each tick so high-rate subscriptions aren't
        # starved. spin_once(timeout_sec=0.0) is non-blocking and processes at
        # most one work item, so we loop with a cap.
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.0)
        self.root.after(20, self.spin_ros)
    
    def run(self):
        self.root.mainloop()


def main():
    rclpy.init()
    gui = VTRControlGUI()
    gui.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
