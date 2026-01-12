# Hybrid-A-Star-Planner
#A customized Hybrid A* Path Planner for non-holonomic robots with optimized steering costs
#This project uses the Hybrid A* base implementation from the [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) library by Atsushi Sakai.
#My Contributions:
#Optimized the cost function logic to resolve priority conflicts.
#Added specific kinematic constraints for custom AMR scenarios.
#Fixed bugs related to analytic expansion bypassing reverse penalties.
1. Install dependencies:
   ```bash
   pip install -r requirements.txt


Run the planner:
Bash
python hybrid_a_star.py
