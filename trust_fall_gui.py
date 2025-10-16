#!/usr/bin/env python3
"""
PyQt GUI for Trust Fall Force Distribution Analysis
"""

import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QGroupBox, QFormLayout, QFileDialog,
    QMessageBox
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

        # Create input parameters group
        params_group = QGroupBox("Simulation Parameters")
        params_layout = QFormLayout()

        # Create input fields with validators
        double_validator = QDoubleValidator(0.0, 10000.0, 2)

        self.person_mass_input = QLineEdit("250")
        self.person_mass_input.setValidator(double_validator)
        params_layout.addRow("Person Mass (lbs):", self.person_mass_input)

        self.person_height_input = QLineEdit("2.0")
        self.person_height_input.setValidator(double_validator)
        params_layout.addRow("Person Height (m):", self.person_height_input)

        self.platform_height_input = QLineEdit("1.5")
        self.platform_height_input.setValidator(double_validator)
        params_layout.addRow("Platform Height (m):", self.platform_height_input)

        self.catch_height_input = QLineEdit("1.219")
        self.catch_height_input.setValidator(double_validator)
        params_layout.addRow("Catch Height (m):", self.catch_height_input)

        self.decel_distance_input = QLineEdit("0.3")
        self.decel_distance_input.setValidator(double_validator)
        params_layout.addRow("Deceleration Distance (m):", self.decel_distance_input)

        self.timing_variance_input = QLineEdit("0.05")
        self.timing_variance_input.setValidator(double_validator)
        params_layout.addRow("Timing Variance (s):", self.timing_variance_input)

        # Computed fall distance display
        self.fall_distance_label = QLabel("0.281 m")
        self.fall_distance_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        params_layout.addRow("Fall Distance (computed):", self.fall_distance_label)

        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        # Update fall distance when heights change
        self.platform_height_input.textChanged.connect(self.update_fall_distance)
        self.catch_height_input.textChanged.connect(self.update_fall_distance)

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
            fall_distance = platform_height - catch_height
            self.fall_distance_label.setText(f"{fall_distance:.3f} m")

            if fall_distance < 0:
                self.fall_distance_label.setStyleSheet("font-weight: bold; color: red;")
            else:
                self.fall_distance_label.setStyleSheet("font-weight: bold; color: #0066cc;")
        except ValueError:
            self.fall_distance_label.setText("Invalid input")
            self.fall_distance_label.setStyleSheet("font-weight: bold; color: red;")

    def run_simulation(self):
        """Run the trust fall simulation with current parameters"""
        try:
            # Parse input values
            person_mass_lbs = float(self.person_mass_input.text())
            person_mass_kg = person_mass_lbs * 0.453592
            person_height = float(self.person_height_input.text())
            platform_height = float(self.platform_height_input.text())
            catch_height = float(self.catch_height_input.text())
            fall_distance = platform_height - catch_height
            decel_distance = float(self.decel_distance_input.text())
            timing_variance = float(self.timing_variance_input.text())

            # Validate inputs
            if fall_distance <= 0:
                QMessageBox.warning(self, "Invalid Input",
                                  "Fall distance must be positive (platform must be higher than catch height)")
                return

            if any(v <= 0 for v in [person_mass_lbs, person_height, decel_distance]):
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
            self.figure = visualize_results(sim, optimized_results)

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

            QMessageBox.information(self, "Success",
                                  f"Simulation complete!\n\n"
                                  f"Optimal catchers: {optimal_catchers}\n"
                                  f"Impact velocity: {sim.impact_velocity:.2f} m/s\n"
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
