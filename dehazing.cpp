#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>


namespace Dehazing {

const int MAX_VALUE{255};
const int DEFAULT_PATCH_SIZE{15}; // Tem que ser um número ímpar

inline unsigned char min(const cv::Vec3b &pixel) {
    unsigned char m = pixel[0] < pixel[1] ? pixel[0] : pixel[1];
    return m < pixel[2] ? m : pixel[2];
}

cv::Mat dark_channel(const cv::Mat &image, const int patch_size) {
    const cv::Size size = image.size();
    const int width = size.width;
    const int height = size.height;

    cv::Mat background = cv::Mat::zeros(size, CV_8UC1);
    cv::Mat dark_channel = cv::Mat::zeros(size, CV_8UC1);
    dark_channel = cv::Scalar::all(MAX_VALUE);

    // Para cada pixel, vamos capturar o valor do canal mais baixo.
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            background.at<unsigned char>(i, j) = min(image.at<cv::Vec3b>(i, j));
        }
    }

    //std::string background_window = "Background";
    //cv::namedWindow(background_window);
    //cv::imshow(background_window, background);

    double min_val{0}, max_val{0}; // O valor máximo não será usado.

    int patch_center = (patch_size - 1) / 2;
    /* Percorre os "patches" para determinar os valores mínimos */
    for (int i = patch_center; i < (height - patch_center); ++i) {
        for (int j = patch_center; j < (width - patch_center); ++j) {
            // "patch" de raio 15
            cv::Mat patch = background
                .colRange(j - patch_center, j + patch_center + 1)
                .rowRange(i - patch_center, i + patch_center + 1);

            // Calcula o mínimo valor dentro do "patch"
            cv::minMaxLoc(patch, &min_val, &max_val);

            // Preenche o patch do "dark channel" com o valor mínimo
            dark_channel
                .colRange(j - patch_center, j + patch_center + 1)
                .rowRange(i - patch_center, i + patch_center + 1) = cv::Scalar::all(min_val);
        }
    }

    return std::move(dark_channel);
}

    // A luz atmosférica é a cor composta pelos maiores valores
    // de cada canal encontrados na imagem
    cv::Scalar atmospheric_light(const cv::Mat image) {
        cv::Size s = image.size();
        const int width = s.width;
        const int height = s.height;

        double blue = 0;
        double green = 0;
        double red = 0;

        // O array contendo os bytes da imagem.
        // Cada pixel é representado por três posições consecutivas do array.
        const unsigned char *data = image.data;

        const int n_channels = 3;
        const int length = width * n_channels * height;
        for (int i = 0; i < length; i += n_channels) {
            if (blue < data[i]) {
                blue = data[i];
            }
            if (green < data[i + 1]) {
                green = data[i + 1];
            }
            if (red < data[i + 2]) {
                red = data[i + 2];
            }
        }

        return std::move(cv::Scalar(blue, green, red));
    }


    double max(const double &a, const double &b, const double &c) {
        if (a > b && a > c) {
            return a;
        } else if (b > c) {
            return b;
        } else {
            return c;
        }
    }

    // O que é transmissão??
    // Transmission Light é a porção de luz que não é dispersada e que chega até a câmera
    cv::Mat transmission(const cv::Mat &dark_channel, const cv::Scalar atmospheric_light, const double haze_amount = 0.05) {
        const double original_value = 1.0 - haze_amount;
        const double max_atmospheric_light = max(atmospheric_light[0], atmospheric_light[1],  atmospheric_light[2]);

        const cv::Size size = dark_channel.size();
        cv::Mat transmission = cv::Mat::zeros(size, CV_8UC1);

        const int width = size.width;
        const int height = size.height;
        unsigned char pixel;

        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                transmission.at<unsigned char>(i, j) = (1.0 - original_value * dark_channel.at<unsigned char>(i, j) /
                        max_atmospheric_light) * MAX_VALUE;
            }
        }

        return std::move(transmission);
    }

    cv::Mat dehazing(const cv::Mat image, const cv::Mat transmission, const cv::Scalar atmospheric_light) {
        cv::Size size = image.size();
        cv::Mat output = cv::Mat::zeros(size, image.type());
        cv::Scalar transmission_pixel;
        cv::Vec3b image_pixel;
        const int width = size.width;
        const int height = size.height;
        const double threshold = 0.1;

        double fraction;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                transmission_pixel = transmission.at<uchar>(i, j);
                image_pixel = image.at<cv::Vec3b>(i, j);

                fraction = transmission_pixel[0] / MAX_VALUE;
                transmission_pixel[0] = fraction < threshold ? threshold : fraction;

                auto &ouput_pixel = output.at<cv::Vec3b>(i, j);
                ouput_pixel[0] = (image_pixel[0] - atmospheric_light[0]) / transmission_pixel[0] + atmospheric_light[0];
                ouput_pixel[1] = (image_pixel[1] - atmospheric_light[1]) / transmission_pixel[0] + atmospheric_light[1];
                ouput_pixel[2] = (image_pixel[2] - atmospheric_light[2]) / transmission_pixel[0] + atmospheric_light[2];
            }
        }

        return std::move(output);
    }
}


int main() {
    std::string original_image_window = "Original Image";
    std::string dark_channel_window = "Dark Channel";
    std::string transmission_window = "Transmission";
    std::string result_window = "Result";

    cv::Mat original_image = cv::imread("image2.jpeg");

    cv::namedWindow(original_image_window, CV_LOAD_IMAGE_COLOR);
    cv::namedWindow(dark_channel_window, CV_LOAD_IMAGE_GRAYSCALE);
    cv::namedWindow(transmission_window, CV_LOAD_IMAGE_GRAYSCALE);
    cv::namedWindow(result_window, CV_LOAD_IMAGE_GRAYSCALE);

    cv::Mat dark_channel = Dehazing::dark_channel(original_image, Dehazing::DEFAULT_PATCH_SIZE);

    cv::Scalar atmospheric_light = Dehazing::atmospheric_light(original_image);

    // Temos que refinar a matriz de transmissão.
    // resolver matriz laplaciana
    cv::Mat transmission = Dehazing::transmission(dark_channel, atmospheric_light);

    cv::Mat result = Dehazing::dehazing(original_image, transmission, atmospheric_light);

    // TODO: Aumentar a radiância


    cv::imshow(original_image_window, original_image);
    cv::imshow(dark_channel_window, dark_channel);
    cv::imshow(transmission_window, transmission);
    cv::imshow(result_window, result);

    cv::waitKey(0);
    return 0;
}
