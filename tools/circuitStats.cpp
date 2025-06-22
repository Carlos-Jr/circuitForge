#include <iostream>
#include <string>
#include <regex>
#include <cstdio>
#include <memory>
#include <optional>
#include <array>

// Struct para armazenar os resultados
struct CircuitInfo {
    int inputs;
    int outputs;
    int gates;
    int levels;
    double energy;
};

// Função que analisa um único arquivo .v e retorna os valores
std::optional<CircuitInfo> analyze_circuit(const std::string& verilog_file) {
    CircuitInfo info;
    
    // Executa bit-combs
    std::string cmd1 = "./bit-combs -o temp.output " + verilog_file + " 2>&1";
    std::array<char, 128> buffer;
    std::string output1;
    
    FILE* pipe1 = popen(cmd1.c_str(), "r");
    if (!pipe1) return std::nullopt;
    
    while (fgets(buffer.data(), buffer.size(), pipe1)) {
        output1 += buffer.data();
    }
    
    if (pclose(pipe1) != 0) return std::nullopt;
    
    // Extrai valores com regex
    std::regex input_regex(R"(inputs\s*=\s*([0-9]+))");
    std::regex output_regex(R"(outputs\s*=\s*([0-9]+))");
    std::regex gates_regex(R"(gates\s*=\s*([0-9]+))");
    std::regex levels_regex(R"(levels\s*=\s*([0-9]+))");
    
    std::smatch match;
    
    if (!std::regex_search(output1, match, input_regex)) return std::nullopt;
    info.inputs = std::stoi(match[1]);
    
    if (!std::regex_search(output1, match, output_regex)) return std::nullopt;
    info.outputs = std::stoi(match[1]);
    
    if (!std::regex_search(output1, match, gates_regex)) return std::nullopt;
    info.gates = std::stoi(match[1]);
    
    if (!std::regex_search(output1, match, levels_regex)) return std::nullopt;
    info.levels = std::stoi(match[1]);
    
    // Executa join-combs
    std::string cmd2 = "./join-combs temp.output 2>&1";
    std::string output2;
    
    FILE* pipe2 = popen(cmd2.c_str(), "r");
    if (!pipe2) return std::nullopt;
    
    while (fgets(buffer.data(), buffer.size(), pipe2)) {
        output2 += buffer.data();
    }
    pclose(pipe2);
    
    // Extrai último valor de energy
    std::regex energy_regex(R"(energy\s*=\s*([0-9.]+))");
    std::string last_energy;
    
    auto begin = std::sregex_iterator(output2.begin(), output2.end(), energy_regex);
    auto end = std::sregex_iterator();
    
    for (auto it = begin; it != end; ++it) {
        last_energy = (*it)[1];  // Sempre pega o último valor
    }
    
    if (last_energy.empty()) return std::nullopt;
    
    info.energy = std::stod(last_energy);
    
    // Remove arquivo temporário
    std::remove("temp.output");
    
    return info;
}

// Exemplo de uso
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <arquivo.v>" << std::endl;
        return 1;
    }
    
    auto result = analyze_circuit(argv[1]);
    
    if (result) {
        std::cout << "Inputs: " << result->inputs << std::endl;
        std::cout << "Outputs: " << result->outputs << std::endl;
        std::cout << "Gates: " << result->gates << std::endl;
        std::cout << "Levels: " << result->levels << std::endl;
        std::cout << "Energy: " << result->energy << std::endl;
    } else {
        std::cerr << "Erro ao processar o arquivo" << std::endl;
        return 1;
    }
    
    return 0;
}