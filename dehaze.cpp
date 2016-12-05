#include <opencv2/opencv.hpp>
#include <string>


// GuidedFilter foi obtido em:
// https://github.com/atilimcetin/guided-filter
class GuidedFilterImpl;

class GuidedFilter {
public:
    GuidedFilter(const cv::Mat &I, int r, double eps);
    ~GuidedFilter();

    cv::Mat filter(const cv::Mat &p, int depth = -1) const;

private:
    GuidedFilterImpl *impl_;
};

cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int depth = -1);

static cv::Mat boxfilter(const cv::Mat &I, int r) {
    cv::Mat result;
    cv::blur(I, result, cv::Size(r, r));
    return result;
}

static cv::Mat convertTo(const cv::Mat &mat, int depth) {
    if (mat.depth() == depth) {
        return mat;
    }

    cv::Mat result;
    mat.convertTo(result, depth);
    return result;
}

class GuidedFilterImpl {
public:
    virtual ~GuidedFilterImpl() {}

    cv::Mat filter(const cv::Mat &p, int depth);

protected:
    int Idepth;

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const = 0;
};

class GuidedFilterMono : public GuidedFilterImpl {
public:
    GuidedFilterMono(const cv::Mat &I, int r, double eps);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
    int r;
    double eps;
    cv::Mat I, mean_I, var_I;
};

class GuidedFilterColor : public GuidedFilterImpl {
public:
    GuidedFilterColor(const cv::Mat &I, int r, double eps);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:
    std::vector<cv::Mat> Ichannels;
    int r;
    double eps;
    cv::Mat mean_I_r, mean_I_g, mean_I_b;
    cv::Mat invrr, invrg, invrb, invgg, invgb, invbb;
};


cv::Mat GuidedFilterImpl::filter(const cv::Mat &p, int depth) {
    cv::Mat p2 = convertTo(p, Idepth);

    cv::Mat result;
    if (p.channels() == 1) {
        result = filterSingleChannel(p2);
    }
    else {
        std::vector<cv::Mat> pc;
        cv::split(p2, pc);

        for (std::size_t i = 0; i < pc.size(); ++i) {
            pc[i] = filterSingleChannel(pc[i]);
        }

        cv::merge(pc, result);
    }

    return convertTo(result, depth == -1 ? p.depth() : depth);
}

GuidedFilterMono::GuidedFilterMono(const cv::Mat &origI, int r, double eps) : r(r), eps(eps) {
    if (origI.depth() == CV_32F || origI.depth() == CV_64F) {
        I = origI.clone();
    }
    else {
        I = convertTo(origI, CV_32F);
    }

    Idepth = I.depth();

    mean_I = boxfilter(I, r);
    cv::Mat mean_II = boxfilter(I.mul(I), r);
    var_I = mean_II - mean_I.mul(mean_I);
}

cv::Mat GuidedFilterMono::filterSingleChannel(const cv::Mat &p) const {
    cv::Mat mean_p = boxfilter(p, r);
    cv::Mat mean_Ip = boxfilter(I.mul(p), r);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(
                         mean_p); // this is the covariance of (I, p) in each local patch.

    cv::Mat a = cov_Ip / (var_I + eps); // Eqn. (5) in the paper;
    cv::Mat b = mean_p - a.mul(mean_I); // Eqn. (6) in the paper;

    cv::Mat mean_a = boxfilter(a, r);
    cv::Mat mean_b = boxfilter(b, r);

    return mean_a.mul(I) + mean_b;
}

GuidedFilterColor::GuidedFilterColor(const cv::Mat &origI, int r, double eps) : r(r), eps(eps) {
    cv::Mat I;
    if (origI.depth() == CV_32F || origI.depth() == CV_64F) {
        I = origI.clone();
    }
    else {
        I = convertTo(origI, CV_32F);
    }

    Idepth = I.depth();

    cv::split(I, Ichannels);

    mean_I_r = boxfilter(Ichannels[0], r);
    mean_I_g = boxfilter(Ichannels[1], r);
    mean_I_b = boxfilter(Ichannels[2], r);

    // variance of I in each local patch: the matrix Sigma in Eqn (14).
    // Note the variance in each local patch is a 3x3 symmetric matrix:
    //           rr, rg, rb
    //   Sigma = rg, gg, gb
    //           rb, gb, bb
    cv::Mat var_I_rr = boxfilter(Ichannels[0].mul(Ichannels[0]), r) - mean_I_r.mul(mean_I_r) + eps;
    cv::Mat var_I_rg = boxfilter(Ichannels[0].mul(Ichannels[1]), r) - mean_I_r.mul(mean_I_g);
    cv::Mat var_I_rb = boxfilter(Ichannels[0].mul(Ichannels[2]), r) - mean_I_r.mul(mean_I_b);
    cv::Mat var_I_gg = boxfilter(Ichannels[1].mul(Ichannels[1]), r) - mean_I_g.mul(mean_I_g) + eps;
    cv::Mat var_I_gb = boxfilter(Ichannels[1].mul(Ichannels[2]), r) - mean_I_g.mul(mean_I_b);
    cv::Mat var_I_bb = boxfilter(Ichannels[2].mul(Ichannels[2]), r) - mean_I_b.mul(mean_I_b) + eps;

    // Inverse of Sigma + eps * I
    invrr = var_I_gg.mul(var_I_bb) - var_I_gb.mul(var_I_gb);
    invrg = var_I_gb.mul(var_I_rb) - var_I_rg.mul(var_I_bb);
    invrb = var_I_rg.mul(var_I_gb) - var_I_gg.mul(var_I_rb);
    invgg = var_I_rr.mul(var_I_bb) - var_I_rb.mul(var_I_rb);
    invgb = var_I_rb.mul(var_I_rg) - var_I_rr.mul(var_I_gb);
    invbb = var_I_rr.mul(var_I_gg) - var_I_rg.mul(var_I_rg);

    cv::Mat covDet = invrr.mul(var_I_rr) + invrg.mul(var_I_rg) + invrb.mul(var_I_rb);

    invrr /= covDet;
    invrg /= covDet;
    invrb /= covDet;
    invgg /= covDet;
    invgb /= covDet;
    invbb /= covDet;
}

cv::Mat GuidedFilterColor::filterSingleChannel(const cv::Mat &p) const {
    cv::Mat mean_p = boxfilter(p, r);

    cv::Mat mean_Ip_r = boxfilter(Ichannels[0].mul(p), r);
    cv::Mat mean_Ip_g = boxfilter(Ichannels[1].mul(p), r);
    cv::Mat mean_Ip_b = boxfilter(Ichannels[2].mul(p), r);

    // covariance of (I, p) in each local patch.
    cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
    cv::Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
    cv::Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);

    cv::Mat a_r = invrr.mul(cov_Ip_r) + invrg.mul(cov_Ip_g) + invrb.mul(cov_Ip_b);
    cv::Mat a_g = invrg.mul(cov_Ip_r) + invgg.mul(cov_Ip_g) + invgb.mul(cov_Ip_b);
    cv::Mat a_b = invrb.mul(cov_Ip_r) + invgb.mul(cov_Ip_g) + invbb.mul(cov_Ip_b);

    cv::Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(
                    mean_I_b); // Eqn. (15) in the paper;

    return (boxfilter(a_r, r).mul(Ichannels[0])
            + boxfilter(a_g, r).mul(Ichannels[1])
            + boxfilter(a_b, r).mul(Ichannels[2])
            + boxfilter(b, r));  // Eqn. (16) in the paper;
}


GuidedFilter::GuidedFilter(const cv::Mat &I, int r, double eps) {
    CV_Assert(I.channels() == 1 || I.channels() == 3);

    if (I.channels() == 1) {
        impl_ = new GuidedFilterMono(I, 2 * r + 1, eps);
    }
    else {
        impl_ = new GuidedFilterColor(I, 2 * r + 1, eps);
    }
}

GuidedFilter::~GuidedFilter() {
    delete impl_;
}

cv::Mat GuidedFilter::filter(const cv::Mat &p, int depth) const {
    return impl_->filter(p, depth);
}

cv::Mat guidedFilter(const cv::Mat &I, const cv::Mat &p, int r, double eps, int depth) {
    return GuidedFilter(I, r, eps).filter(p, depth);
}

//========================================================================
//  Código próprio começa aqui
//========================================================================
using namespace cv;

Mat get_dark_channel(const Mat &image, const int window_size) {
    const cv::Size size = image.size();
    const int width = size.width;
    const int height = size.height;

    Mat background = Mat::zeros(size, CV_8UC1);
    Mat dark_channel = Mat::zeros(size, CV_8UC1);
    dark_channel = Scalar::all(255);

    // Para cada pixel, vamos capturar o valor do canal mais baixo.
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            auto &pixel = image.at<Vec3b>(i, j);
            // output[i, j] = min(pixel[0], pixel[1], pixel[2]);
            if (pixel[0] < pixel[1] && pixel[0] < pixel[2]) {
                background.at<unsigned char>(i, j) = pixel[0];
            }
            else if (pixel[1] < pixel[2]) {
                background.at<unsigned char>(i, j) = pixel[1];
            }
            else {
                background.at<unsigned char>(i, j) = pixel[2];
            }
        }
    }

    double min_val{0}, max_val{0}; // O valor máximo não será usado.

    int window_center = (window_size - 1) / 2;
    /* Percorre os "patches" para determinar os valores mínimos */
    for (int i = window_center; i < (height - window_center); ++i) {
        for (int j = window_center; j < (width - window_center); ++j) {
            // "patch" de raio 15
            Mat patch = background
                        .colRange(j - window_center, j + window_center + 1)
                        .rowRange(i - window_center, i + window_center + 1);

            // Calcula o mínimo valor dentro do "patch"
            minMaxLoc(patch, &min_val, &max_val);

            // Preenche o patch do "dark channel" com o valor mínimo
            dark_channel
            .colRange(j - window_center, j + window_center + 1)
            .rowRange(i - window_center, i + window_center + 1) = Scalar::all(min_val);
        }
    }

    return std::move(dark_channel);
}

Scalar get_atmosphere(const Mat &image) {
    double blue, green, red;
    blue = green = red = 0;

    Size s(image.size());
    const int width = s.width;
    const int height = s.height;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            auto &pixel = image.at<Vec3b>(i, j);
            if (pixel[0] > blue) {
                blue = pixel[0];
            }
            if (pixel[1] > green) {
                green = pixel[1];
            }
            if (pixel[2] > red) {
                red = pixel[2];
            }
        }
    }

    return std::move(Scalar(blue, green, red));
}

// 1 - omega * dark_channel(I / A, w)
Mat get_transmission(const Mat &dark_channel, const Scalar A, const double omega) {

    double min_A; // O maior canal de luz atmosférica
    if (A[0] < A[1] && A[0] < A[2]) {
        min_A = A[0];
    }
    else if (A[1] < A[2]) {
        min_A = A[1];
    }
    else {
        min_A = A[2];
    }

    const cv::Size size = dark_channel.size();
    cv::Mat transmission = cv::Mat::zeros(size, CV_8UC1);

    const int width = size.width;
    const int height = size.height;
    unsigned char pixel;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            transmission.at<unsigned char>(i, j) = (1.0 - omega * dark_channel.at<unsigned char>(i,
                                                    j) / min_A) * 255;
        }
    }

    return std::move(transmission);
}

Mat get_radiance(const Mat &image, const Scalar A, const Mat transmission) {
    Size size(image.size());
    const int width = size.width;
    const int height = size.height;

    Mat output = Mat::zeros(size, image.type());

    const double max_value = 255;
    const double min_value = 0.1;

    Scalar t_p;
    double fraction;
    int blue, green, red;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            auto &in = image.at<Vec3b>(i, j);
            auto &out = output.at<Vec3b>(i, j);
            t_p = transmission.at<unsigned char>(i, j);

            fraction = t_p.val[0] / max_value;
            if (fraction < min_value) {
                t_p.val[0] = min_value;
            } else {
                t_p.val[0] = fraction;
            }

            blue = (in[0] - A[0]) / t_p.val[0] + A[0];
            green = (in[1] - A[1]) / t_p.val[0] + A[1];
            red = (in[2] - A[2]) / t_p.val[0] + A[2];
            out[0] = blue > 255 ? 255 : blue < 0 ? 0 : (uchar) blue;
            out[1] = green > 255 ? 255 : green < 0 ? 0 : (uchar) green;
            out[2] = red > 255 ? 255 : red < 0 ? 0 : (uchar) red;
        }
    }

    return std::move(output);
}

int main(int argc, char **argv) {
    const int w = 15; // O tamanho de cada patch para cálculo do dark channel
    const int A_max = 220; // Threshold da luz atmosférica
    const double omega = 0.95; // "bias" para estimar o meio de tranmissão
    const int r = 40; // Raio default para o filtro
    const double eps = 1e-3; // epsilon default para o filtro

    std::string filename{"test0.jpg"};
    if (argc > 0) {
        filename = argv[1];
    }

    std::string original_image_window{"Original Image"};
    std::string dark_channel_window{"Dark channel"};
    std::string transmission_window{"Transmission"};
    std::string refined_transmission_window{"Refined Transmission"};
    std::string result_window{"Result"};

    namedWindow(original_image_window);
    namedWindow(dark_channel_window);
    namedWindow(transmission_window);
    namedWindow(refined_transmission_window);
    namedWindow(result_window);

    // Lê a imagem
    Mat original_image = imread("img/" + filename);
    imshow(original_image_window, original_image);

    // Obtém o dark channel
    Mat dark_channel = get_dark_channel(original_image, w);
    imwrite("img/output_dark_channel_" + filename, dark_channel);
    imshow(dark_channel_window, dark_channel);

    // Estima a luz atmosférica global
    Scalar A = get_atmosphere(original_image);
    for (int i = 0; i < 3; ++i) {
        if (A[i] > A_max) {
            A[i] = A_max;
        }
    }

    // Estima a transmissão
    Mat transmission = get_transmission(dark_channel, A, omega);
    Mat min_transmission = Mat::zeros(transmission.size(), transmission.type());
    double t_min = 255.0 * 0.2;
    min_transmission = Scalar::all(t_min);
    max(transmission, min_transmission, min_transmission);
    imwrite("img/output_transmission_" + filename, min_transmission);
    imshow(transmission_window, min_transmission);

    // Refina a transmissão, usando Guided Filter
    //
    // Guided Filter usa uma imagem como referência para filtrar outra,
    // nesse caso é usada a imagem original para suavizar a transmissão
    // sem perder os contornos da imagem original.
    Mat guided_transmission = guidedFilter(original_image, min_transmission, r, eps);
    imwrite("img/output_refined_transmission_" + filename, guided_transmission);
    imshow(refined_transmission_window, guided_transmission);

    // Obtém a radiância da cena
    Mat result = get_radiance(original_image, A, guided_transmission);
    imwrite("img/output_" + filename, result);
    imshow(result_window, result);

    waitKey(0);
    return 0;
}

