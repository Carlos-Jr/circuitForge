#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "Erro ao obter contagem de dispositivos: " 
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::cout << "Número de dispositivos CUDA encontrados: " 
              << deviceCount << std::endl;

    if (deviceCount > 0) {
        for (int dev = 0; dev < deviceCount; dev++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, dev);
            
            std::cout << "\nDispositivo " << dev << ":" << std::endl;
            std::cout << "Nome: " << prop.name << std::endl;
            std::cout << "Multiprocessadores: " << prop.multiProcessorCount << std::endl;
            std::cout << "Cores por multiprocessador: " << prop.maxThreadsPerMultiProcessor << std::endl;
        }
    }

    return 0;
}