#include <vector>
#include <iostream>

#include <torch/script.h>
#include <torch/torch.h>

std::vector<c10::IValue> runGPU(torch::jit::script::Module &generator, torch::Tensor &noise, torch::Tensor &energy,
                                const int number, const int batchSize, const int EMin, const int EMax) {

    torch::NoGradGuard no_grad;

    std::vector<c10::IValue> images = {};

    for (int i = 0; i < number; i += batchSize) {

        noise.uniform_(-1, 1);
        energy.uniform_(EMin, EMax);
        auto res = generator.forward({noise, energy});
        res.toTensor().cpu();
        images.emplace_back(res);
        printf("[%7d | %7d]\r", i + batchSize, number);
    }
    return images;

}

std::vector<c10::IValue> runCPU(torch::jit::script::Module &generator, torch::Tensor &noise,
                                torch::Tensor &energy,
                                const int number, const int batchSize, const int EMin, const int EMax) {

    torch::NoGradGuard no_grad;
    std::vector<c10::IValue> images = {};

    for (int i = 0; i < number; i += batchSize) {

        noise.uniform_(-1, 1);
        energy.uniform_(EMin, EMax);
        auto res = generator.forward({noise, energy});
        res.toTensor();
        images.emplace_back(res);
        printf("[%7d | %7d]\r", i + batchSize, number);
        printf ("%ld", res.toTensor().dim());

    }
    return images;

}

int main(int argc, const char *argv[]) {
    if (argc != 7) {
        std::cerr << "usage: <device> <number> <batchSize> <EMin> <EMax> <Path to model> .\n";
        return EXIT_FAILURE;
    }

    if (strcmp(argv[1], "cuda") != 0 && strcmp(argv[1], "cpu") != 0) {
        std::cerr << "Please choose either 'cuda' or 'cpu'." << std::endl;
        return EXIT_FAILURE;
    }

    torch::Device device = (strcmp(argv[1], "cuda") == 0) && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    if (strcmp(argv[1], "cuda") == 0) assert(torch::cuda::is_available());

//  Load the generator as saved in `python/SavePytorchModel.ipynb`
    torch::jit::script::Module generator;

    try {
        generator = torch::jit::load(argv[6]);
    }
    catch (const c10::Error &e) {
        std::cerr << "error loading the model\n";
        return EXIT_FAILURE;
    }

    generator.to(device);
    generator.eval();

//  Our model generates `number` calorimeter showers in batches of `batchSize`
//  corresponding to energies between EMin and EMax.
    int number = atoi(argv[2]);
    int batchSize = atoi(argv[3]);
    int EMin = atoi(argv[4]);
    int EMax = atoi(argv[5]);
    const int latent_dim = 100;

    auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(device);

    torch::Tensor noise = torch::zeros({batchSize, latent_dim, 1, 1, 1}, options);
    torch::Tensor energy = torch::zeros({batchSize, 1, 1, 1, 1}, options);

//  Nothing is done here with the generated showers, feel free to adapt to your needs.
    if (device.type() == torch::kCUDA) runGPU(generator, noise, energy, number, batchSize, EMin, EMax);
    else runCPU(generator, noise, energy, number, batchSize, EMin, EMax);
    return EXIT_SUCCESS;
}