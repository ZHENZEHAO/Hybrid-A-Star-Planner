# Hybrid A* Path Planner

## 🎥 Project Demo
![Demo Animation](demo_one.gif)

## 🎥 Project Demo
![Demo Animation](demo_two.gif)

## 📖 Overview
This project implements the **Hybrid A* algorithm** for non-holonomic robots (Ackermann steering). It optimizes path generation by resolving conflicts between heuristic search and analytic expansion (Reeds-Shepp curves).

## 🚀 Key Features
- **Kinematic Constraints:** Enforces minimum turning radius for realistic vehicle movement.
- **Smoother Trajectories:** Added steering cost and steering rate cost to prevent path oscillation.
- **Improved Goal Logic:** Checks both Position and Orientation (Yaw) at the goal.

## 🛠️ How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
