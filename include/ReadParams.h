#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>
#include <unordered_map>

std::unordered_map<std::string, double> readParameters(const std::filesystem::path& filePath) {
    std::unordered_map<std::string, double> parameters;
    std::ifstream file(filePath);
    std::string line;

    if (!file.is_open()) {
        std::cout << "Unable to open file: " << filePath << std::endl;
        return parameters;
    }

    while (std::getline(file, line)) {

        // Skip lines that start with '#'
        if (!line.empty() && line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string name;
        char equalSign;
        double value;

        if (iss >> name >> equalSign >> value) {
            if (equalSign == '=') {
                parameters[name] = value;
            } else {
                std::cout << "Invalid format in line: " << line << std::endl;
            }
        } 
        else {
            std::cout << "Invalid line: " << line << std::endl;
        }
    }

    file.close();
    return parameters;
}