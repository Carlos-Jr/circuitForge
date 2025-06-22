#include <iostream>
#include <filesystem>
#include <string>
#include <regex>
#include <cstdio>
#include <memory>
#include <array>

// Função principal que analisa uma pasta com arquivos .v
void analyze_verilog_folder(const std::string& folder_path) {
    // Cabeçalho CSV
    std::cout << "arquivo,inputs,outputs,gates,levels,energy" << std::endl;
    
    // Itera sobre arquivos .v na pasta
    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".v") {
            continue;
        }
        
        std::string filename = entry.path().stem().string();
        std::string filepath = entry.path().string();
        
        // Executa bit-combs
        std::string cmd1 = "./bit-combs -o temp.output " + filepath + " 2>&1";
        std::array<char, 128> buffer;
        std::string output1;
        
        FILE* pipe1 = popen(cmd1.c_str(), "r");
        if (!pipe1) continue;
        
        while (fgets(buffer.data(), buffer.size(), pipe1)) {
            output1 += buffer.data();
        }
        
        if (pclose(pipe1) != 0) continue;
        
        // Extrai valores com regex
        std::regex input_regex(R"(inputs\s*=\s*([0-9]+))");
        std::regex output_regex(R"(outputs\s*=\s*([0-9]+))");
        std::regex gates_regex(R"(gates\s*=\s*([0-9]+))");
        std::regex levels_regex(R"(levels\s*=\s*([0-9]+))");
        
        std::smatch match;
        std::string inputs, outputs, gates, levels;
        
        if (std::regex_search(output1, match, input_regex)) inputs = match[1];
        if (std::regex_search(output1, match, output_regex)) outputs = match[1];
        if (std::regex_search(output1, match, gates_regex)) gates = match[1];
        if (std::regex_search(output1, match, levels_regex)) levels = match[1];
        
        if (inputs.empty() || outputs.empty() || gates.empty() || levels.empty()) {
            continue;
        }
        
        // Executa join-combs
        std::string cmd2 = "./join-combs temp.output 2>&1";
        std::string output2;
        
        FILE* pipe2 = popen(cmd2.c_str(), "r");
        if (!pipe2) continue;
        
        while (fgets(buffer.data(), buffer.size(), pipe2)) {
            output2 += buffer.data();
        }
        pclose(pipe2);
        
        // Extrai último valor de energy
        std::regex energy_regex(R"(energy\s*=\s*([0-9.]+))");
        std::string energy;
        
        auto begin = std::sregex_iterator(output2.begin(), output2.end(), energy_regex);
        auto end = std::sregex_iterator();
        
        for (auto it = begin; it != end; ++it) {
            energy = (*it)[1];  // Sempre pega o último valor
        }
        
        if (!energy.empty()) {
            // Imprime linha CSV
            std::cout << filename << "," << inputs << "," << outputs << "," 
                     << gates << "," << levels << "," << energy << std::endl;
        }
    }
    
    // Remove arquivo temporário
    std::filesystem::remove("temp.output");
}

// Função main
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Uso: " << argv[0] << " <caminho_para_pasta>" << std::endl;
        return 1;
    }
    
    analyze_verilog_folder(argv[1]);
    return 0;
}