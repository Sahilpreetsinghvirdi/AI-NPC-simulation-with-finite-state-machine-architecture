# AI NPC Simulation

A modular C++20 simulation sandbox for developing intelligent NPC behavior for large-scale open-world games.

This project serves as an experimental environment for testing AI systems such as police pursuit, player escape behavior, reinforcement learning integration, state machines, perception, and decision-making before integrating them into a complete game.

---

## Features

- Modular C++20 architecture
- CMake build system
- Police AI state machine
- Dynamic wanted level system
- Multiple police officers
- Health and combat system
- Continuous steering movement
- Local boundary perception
- Experience-based escape memory
- Scrollable debug panel
- Real-time simulation visualization
- Reinforcement Learning ready architecture

---

## Current AI Systems

### Player

- Dynamic escape behavior
- Local obstacle perception
- Boundary awareness
- Escape decision logic
- Health system
- Wanted level system

### Police

- Patrol
- Investigate
- Pursue
- Attack
- Dynamic speed scaling
- Multi-officer pursuit
- State machine architecture

---

## Planned Features

- Reinforcement Learning (PPO)
- ONNX Runtime integration
- Experience replay
- Neural network policies
- Pathfinding (A*)
- Navigation mesh
- Vehicle AI
- Civilian AI
- Economy simulation
- Traffic system
- Large world simulation
- Multiplayer synchronization

---

## Technology

- C++20
- CMake
- Visual Studio 2022/2026
- ONNX Runtime (planned)
- Python (planned RL training)
- PPO Reinforcement Learning (planned)

---

## Project Structure

```
ai-npc-simulation/
│
├── include/
├── src/
├── data/
├── models/
├── logs/
├── tests/
└── CMakeLists.txt
```

---

## Build

```bash
git clone <repo>

cd ai-npc-simulation

cmake -S . -B build

cmake --build build
```

Run:

```bash
./build/bin/ai_npc_sim
```

---

## Roadmap

- [x] Simulation loop
- [x] Player system
- [x] Police AI
- [x] Wanted level
- [x] Combat system
- [x] Dynamic police spawning
- [x] Steering behaviors
- [x] Local perception
- [x] Experience memory
- [x] Debug interface
- [ ] Reinforcement Learning pipeline
- [ ] ONNX integration
- [ ] Pathfinding
- [ ] Navigation system
- [ ] Vehicles
- [ ] Traffic
- [ ] Large-scale NPC simulation

---

## Vision

The long-term goal of this project is to build scalable AI systems capable of supporting persistent, intelligent NPCs for a large open-world simulation.

Rather than focusing on graphics first, the project prioritizes simulation, decision-making, learning, and extensible AI architecture.

---

## License

This project is currently under active development.

![AI NPC Simulation Screenshot](Screenshot%202026-07-03%20160046.png)