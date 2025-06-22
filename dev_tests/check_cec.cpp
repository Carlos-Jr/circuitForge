#include <iostream>
#include <string>
#include <cstdio>
#include <regex>
#include <fstream>
#include <array>

int check_equivalence(const std::string& circuit1_file, const std::string& circuit2_file) {
    std::string cmd = "yosys-abc -c \"read " + circuit1_file + "; " +
                      "cec " + circuit2_file + "; " +
                      "quit\" 2>&1";
    
    std::array<char, 128> buffer;
    std::string output;
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        return 0; // Erro ao executar ABC
    }
    
    while (fgets(buffer.data(), buffer.size(), pipe)) {
        output += buffer.data();
    }
    pclose(pipe);
    
    if (output.find("Networks are equivalent") != std::string::npos) {
        return 1;
    }
    
    return 0;
}

// Exemplo de uso
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Uso: " << argv[0] << " <circuito1.v> <circuito2.v>" << std::endl;
        return 1;
    }
    
    std::string circuit1 = argv[1];
    std::string circuit2 = argv[2];
    
    std::cout << "Verificando equivalência entre:" << std::endl;
    std::cout << "  Circuito 1: " << circuit1 << std::endl;
    std::cout << "  Circuito 2: " << circuit2 << std::endl;
    std::cout << std::endl;
    
    auto result = check_equivalence(circuit1, circuit2);
    std::cout << "Resultado: " << (int)result << std::endl;
    
    return 0;
}