# CircuitForge: Advanced Digital Circuit Enhancement System

An intelligent system for digital circuit analysis and optimization using BDD-based entropy calculation and equivalence checking for superior circuit design enhancement.

## 🚀 Overview

CircuitForge leverages cutting-edge tools to analyze, optimize, and enhance digital circuits through entropy-based analysis and automated verification. The system integrates **trope**, a BDD-based circuit entropy calculator, with **ABC** synthesis and verification capabilities to provide comprehensive circuit optimization solutions.

## 🛠️ Core Technologies

### Trope
- **Description**: BDD-based circuit's entropy calculator
- **Repository**: https://github.com/jefchaves/trope
- **Purpose**: Circuit entropy calculation and digital circuit analysis

### ABC
- **Description**: System for Sequential Synthesis and Verification
- **Repository**: https://github.com/berkeley-abc/abc
- **Purpose**: Circuit synthesis and equivalence verification

## 📁 Project Structure

### `tools/` Directory

Contains pre-compiled **Linux x86-64** binaries for the trope software suite:

#### Core Binaries
- **`app-energy`**: Main application for energy consumption calculation
- **`bit-combs`**: Bit combination analysis tool
- **`join-combs`**: Combination joining and processing tool

#### `csvFolder` Utility
- **Function**: Batch processing tool that utilizes `bit-combs` and `join-combs`
- **Input**: Directory containing Verilog files (.v)
- **Output**: CSV report with comprehensive circuit metrics:
  - **inputs**: Number of circuit inputs
  - **outputs**: Number of circuit outputs
  - **gates**: Total gate count
  - **levels**: Logic depth levels
  - **energy**: Calculated energy consumption

### `dev_tests/` Directory

Contains individual development tests and experiments conducted during system development.

## 💻 System Requirements

- **Operating System**: Linux x86-64
- **Dependencies**: 
  - **CUDD Library**: Required dependency for trope
    - Repository: https://github.com/ivmai/cudd
    - Installation: Compile and install from source code
  - **ABC (yosys-abc)**: Circuit synthesis and verification system
    - **From source**: https://github.com/berkeley-abc/abc
    - **Fedora**: `sudo dnf install yosys-abc`
    - **OpenSUSE**: `sudo zypper install yosys-abc`
  - Valid Verilog files for analysis

## 🔧 Usage

### Individual Circuit Analysis
```bash
# Run energy analysis
./tools/circuitStats circuit.v
```

### Batch Processing with csvFolder
```bash
# Process all .v files in a directory
./tools/csvFolder /path/to/verilog/files/directory
```

## ✨ Features

- ✅ BDD-based entropy calculation
- ✅ Energy consumption analysis
- ✅ Batch processing of multiple circuits
- ✅ Automated CSV report generation
- ✅ Circuit equivalence verification
- ✅ Circuit synthesis and optimization
- ✅ Comprehensive circuit metrics analysis

## 🔮 Future Features

- 🔄 **Cartesian Genetic Programming (CGP)**: Integration of evolutionary algorithms for automated circuit design and optimization
- 🧠 **Machine Learning Integration**: AI-driven circuit optimization strategies
- 📊 **Advanced Visualization**: Interactive circuit analysis dashboards
- 🌐 **Web Interface**: Browser-based circuit analysis platform

## 🤝 Contributing

We welcome contributions to CircuitOptimizer! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📋 Installation

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd CircuitOptimizer
   ```

2. **Install CUDD Library**
   ```bash
   git clone https://github.com/ivmai/cudd
   cd cudd
   ./configure --enable-silent-rules --enable-obj --enable-dddmp --enable-shared
   make -j4 check
   sudo make install
   ```

3. **Install ABC**
   - **Fedora**: `sudo dnf install yosys-abc`
   - **OpenSUSE**: `sudo zypper install yosys-abc`
   - **From source**: Follow instructions at https://github.com/berkeley-abc/abc

4. **Make binaries executable**
   ```bash
   chmod +x tools/*
   ```


---

**Note**: The included binaries are compiled specifically for Linux x86-64. For other architectures, recompilation from trope source code will be required.