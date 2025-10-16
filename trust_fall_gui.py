#!/usr/bin/env python3
"""
PyQt GUI for Trust Fall Force Distribution Analysis
"""

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QFormLayout, QFileDialog,
    QMessageBox, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from trust_fall_analysis import TrustFallSimulation, visualize_results


class TrustFallGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trust Fall Force Distribution Analysis")
        self.setGeometry(100, 100, 1200, 900)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Create unit system toggle
        unit_group = QGroupBox("Unit System")
        unit_layout = QHBoxLayout()
        
        self.metric_radio = QRadioButton("Metric")
        self.imperial_radio = QRadioButton("Imperial")
        self.metric_radio.setChecked(True)  # Default to metric
        
        self.unit_group = QButtonGroup()
        self.unit_group.addButton(self.metric_radio)
        self.unit_group.addButton(self.imperial_radio)
        
        unit_layout.addWidget(self.metric_radio)
        unit_layout.addWidget(self.imperial_radio)
        unit_layout.addStretch()
        
        unit_group.setLayout(unit_layout)
        main_layout.addWidget(unit_group)

        # Create input parameters group
        params_group = QGroupBox("Simulation Parameters")
        params_layout = QFormLayout()

        # Store current unit system
        self.is_metric = True

        # Create input fields with validators
        double_validator = QDoubleValidator(0.0, 10000.0, 2)

        # Default values in metric (backend expected units)
        self.person_mass_input = QLineEdit("113.4")  # 250 lbs = 113.4 kg
        self.person_mass_input.setValidator(double_validator)
        self.person_mass_label = QLabel("Person Mass (kg):")
        params_layout.addRow(self.person_mass_label, self.person_mass_input)

        self.person_height_input = QLineEdit("2.0")
        self.person_height_input.setValidator(double_validator)
        self.person_height_label = QLabel("Person Height (m):")
        params_layout.addRow(self.person_height_label, self.person_height_input)

        self.platform_height_input = QLineEdit("1.5")
        self.platform_height_input.setValidator(double_validator)
        self.platform_height_label = QLabel("Platform Height (m):")
        params_layout.addRow(self.platform_height_label, self.platform_height_input)

        self.catch_height_input = QLineEdit("1.219")
        self.catch_height_input.setValidator(double_validator)
        self.catch_height_label = QLabel("Catch Height (m):")
        params_layout.addRow(self.catch_height_label, self.catch_height_input)

        self.decel_distance_input = QLineEdit("0.3")
        self.decel_distance_input.setValidator(double_validator)
        self.decel_distance_label = QLabel("Deceleration Distance (m):")
        params_layout.addRow(self.decel_distance_label, self.decel_distance_input)

        self.timing_variance_input = QLineEdit("0.05")
        self.timing_variance_input.setValidator(double_validator)
        params_layout.addRow("Timing Variance (s):", self.timing_variance_input)

        # Computed fall distance display
        self.fall_distance_label = QLabel("0.281 m")
        self.fall_distance_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        self.fall_distance_display_label = QLabel("Fall Distance (computed):")
        params_layout.addRow(self.fall_distance_display_label, self.fall_distance_label)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # Update fall distance when heights change
        self.platform_height_input.textChanged.connect(self.update_fall_distance)
        self.catch_height_input.textChanged.connect(self.update_fall_distance)
        
        # Connect unit system changes
        self.metric_radio.toggled.connect(self.on_unit_system_changed)
        self.imperial_radio.toggled.connect(self.on_unit_system_changed)

        # Create buttons
        button_layout = QHBoxLayout()

        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 10px; font-size: 14px;")
        button_layout.addWidget(self.run_button)

        self.save_button = QPushButton("Save Visualization")
        self.save_button.clicked.connect(self.save_visualization)
        self.save_button.setEnabled(False)
        self.save_button.setStyleSheet("background-color: #008CBA; color: white; padding: 10px; font-size: 14px;")
        button_layout.addWidget(self.save_button)

        main_layout.addLayout(button_layout)

        # Create matplotlib figure canvas
        self.figure = None
        self.canvas = None
        self.toolbar = None

        # Placeholder for canvas
        self.canvas_placeholder = QWidget()
        self.canvas_placeholder.setMinimumHeight(600)
        main_layout.addWidget(self.canvas_placeholder)

        # Update initial fall distance
        self.update_fall_distance()

    def update_fall_distance(self):
        """Update the computed fall distance"""
        try:
            platform_height = float(self.platform_height_input.text())
            catch_height = float(self.catch_height_input.text())
            
            # Convert to metric if needed for calculation
            if not self.is_metric:
                platform_height = platform_height * 0.3048  # feet to meters
                catch_height = catch_height * 0.3048  # feet to meters
            
            fall_distance = platform_height - catch_height
            
            # Display in current unit system
            if self.is_metric:
                self.fall_distance_label.setText(f"{fall_distance:.3f} m")
                self.fall_distance_display_label.setText("Fall Distance (computed):")
            else:
                fall_distance_ft = fall_distance / 0.3048  # meters to feet
                self.fall_distance_label.setText(f"{fall_distance_ft:.3f} ft")
                self.fall_distance_display_label.setText("Fall Distance (computed):")

            if fall_distance < 0:
                self.fall_distance_label.setStyleSheet("font-weight: bold; color: red;")
            else:
                self.fall_distance_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        except ValueError:
            self.fall_distance_label.setText("Invalid input")
            self.fall_distance_label.setStyleSheet("font-weight: bold; color: red;")

    def on_unit_system_changed(self):
        """Handle unit system change"""
        if self.sender() == self.metric_radio and self.metric_radio.isChecked():
            if not self.is_metric:
                self.convert_to_metric()
                self.is_metric = True
        elif self.sender() == self.imperial_radio and self.imperial_radio.isChecked():
            if self.is_metric:
                self.convert_to_imperial()
                self.is_metric = False
        
        self.update_fall_distance()

    def convert_to_metric(self):
        """Convert all input values from Imperial to Metric"""
        try:
            # Convert mass from lbs to kg
            mass_lbs = float(self.person_mass_input.text())
            mass_kg = mass_lbs * 0.453592
            self.person_mass_input.setText(f"{mass_kg:.1f}")
            self.person_mass_label.setText("Person Mass (kg):")
            
            # Convert heights from feet to meters
            height_ft = float(self.person_height_input.text())
            height_m = height_ft * 0.3048
            self.person_height_input.setText(f"{height_m:.2f}")
            self.person_height_label.setText("Person Height (m):")
            
            platform_ft = float(self.platform_height_input.text())
            platform_m = platform_ft * 0.3048
            self.platform_height_input.setText(f"{platform_m:.3f}")
            self.platform_height_label.setText("Platform Height (m):")
            
            catch_ft = float(self.catch_height_input.text())
            catch_m = catch_ft * 0.3048
            self.catch_height_input.setText(f"{catch_m:.3f}")
            self.catch_height_label.setText("Catch Height (m):")
            
            decel_ft = float(self.decel_distance_input.text())
            decel_m = decel_ft * 0.3048
            self.decel_distance_input.setText(f"{decel_m:.3f}")
            self.decel_distance_label.setText("Deceleration Distance (m):")
            
        except ValueError:
            pass  # Invalid input, don't convert

    def convert_to_imperial(self):
        """Convert all input values from Metric to Imperial"""
        try:
            # Convert mass from kg to lbs
            mass_kg = float(self.person_mass_input.text())
            mass_lbs = mass_kg / 0.453592
            self.person_mass_input.setText(f"{mass_lbs:.0f}")
            self.person_mass_label.setText("Person Mass (lbs):")
            
            # Convert heights from meters to feet
            height_m = float(self.person_height_input.text())
            height_ft = height_m / 0.3048
            self.person_height_input.setText(f"{height_ft:.1f}")
            self.person_height_label.setText("Person Height (ft):")
            
            platform_m = float(self.platform_height_input.text())
            platform_ft = platform_m / 0.3048
            self.platform_height_input.setText(f"{platform_ft:.1f}")
            self.platform_height_label.setText("Platform Height (ft):")
            
            catch_m = float(self.catch_height_input.text())
            catch_ft = catch_m / 0.3048
            self.catch_height_input.setText(f"{catch_ft:.1f}")
            self.catch_height_label.setText("Catch Height (ft):")
            
            decel_m = float(self.decel_distance_input.text())
            decel_ft = decel_m / 0.3048
            self.decel_distance_input.setText(f"{decel_ft:.1f}")
            self.decel_distance_label.setText("Deceleration Distance (ft):")
            
        except ValueError:
            pass  # Invalid input, don't convert

    def run_simulation(self):
        """Run the trust fall simulation with current parameters"""
        try:
            # Parse input values and convert to metric (backend expected units)
            if self.is_metric:
                person_mass_kg = float(self.person_mass_input.text())
                person_height = float(self.person_height_input.text())
                platform_height = float(self.platform_height_input.text())
                catch_height = float(self.catch_height_input.text())
                decel_distance = float(self.decel_distance_input.text())
            else:
                # Convert from Imperial to Metric
                person_mass_lbs = float(self.person_mass_input.text())
                person_mass_kg = person_mass_lbs * 0.453592
                
                person_height_ft = float(self.person_height_input.text())
                person_height = person_height_ft * 0.3048  # feet to meters
                
                platform_height_ft = float(self.platform_height_input.text())
                platform_height = platform_height_ft * 0.3048  # feet to meters
                
                catch_height_ft = float(self.catch_height_input.text())
                catch_height = catch_height_ft * 0.3048  # feet to meters
                
                decel_distance_ft = float(self.decel_distance_input.text())
                decel_distance = decel_distance_ft * 0.3048  # feet to meters
            
            fall_distance = platform_height - catch_height
            timing_variance = float(self.timing_variance_input.text())

            # Validate inputs
            if fall_distance <= 0:
                QMessageBox.warning(self, "Invalid Input",
                                  "Fall distance must be positive (platform must be higher than catch height)")
                return

            if any(v <= 0 for v in [person_mass_kg, person_height, decel_distance]):
                QMessageBox.warning(self, "Invalid Input",
                                  "All parameters must be positive values")
                return

            # Create simulation
            sim = TrustFallSimulation(
                person_mass=person_mass_kg,
                person_height=person_height,
                fall_distance=fall_distance,
                catch_height=catch_height,
                timing_variance=timing_variance
            )

            # Find optimal configuration
            optimal_catchers, optimal_results = sim.find_minimum_catchers(
                max_force_per_catcher=400,
                deceleration_distance=decel_distance
            )

            # Optimize positions
            optimized_positions = sim.optimize_catcher_positions(optimal_catchers, decel_distance)
            optimized_results = sim.simulate_catch(
                optimal_catchers,
                decel_distance,
                optimized_positions
            )

            # Generate visualization
            self.figure = visualize_results(sim, optimized_results, use_metric=self.is_metric)

            # Remove old canvas if it exists
            if self.canvas:
                self.canvas.deleteLater()
                self.toolbar.deleteLater()

            # Create new canvas
            self.canvas = FigureCanvas(self.figure)
            self.toolbar = NavigationToolbar(self.canvas, self)

            # Replace placeholder with canvas
            layout = self.centralWidget().layout()
            layout.removeWidget(self.canvas_placeholder)
            self.canvas_placeholder.deleteLater()

            # Add toolbar and canvas
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)

            # Update placeholder reference
            self.canvas_placeholder = self.canvas

            # Enable save button
            self.save_button.setEnabled(True)

            # Format velocity display based on unit preference
            if self.is_metric:
                velocity_str = f"{sim.impact_velocity:.2f} m/s"
            else:
                velocity_fps = sim.impact_velocity / 0.3048  # Convert m/s to ft/s
                velocity_str = f"{velocity_fps:.1f} ft/s"

            QMessageBox.information(self, "Success",
                                  f"Simulation complete!\n\n"
                                  f"Optimal catchers: {optimal_catchers}\n"
                                  f"Impact velocity: {velocity_str}\n"
                                  f"Deceleration time: {optimized_results['deceleration_time']*1000:.0f} ms")

        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input",
                              f"Please enter valid numeric values\n\nError: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error",
                               f"An error occurred during simulation:\n\n{str(e)}")

    def save_visualization(self):
        """Save the current visualization to a PNG file"""
        if not self.figure:
            QMessageBox.warning(self, "No Visualization",
                              "Please run a simulation first")
            return

        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Visualization",
            "trust_fall_analysis.png",
            "PNG Files (*.png);;All Files (*)"
        )

        if file_path:
            try:
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success",
                                      f"Visualization saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error",
                                   f"Failed to save visualization:\n\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look

    window = TrustFallGUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
