//
// Created by eundolee on 19. 12. 17..
//

#ifndef RS_PCL_TRANSLATION_ESTIMATOR_HPP
#define RS_PCL_TRANSLATION_ESTIMATOR_HPP

enum EAxis {
    EAxisX=0,
    EAxisY,
    EAxisZ
};

class TranslationEstimator {
public:
    Eigen::Translation3f estimate_translation(std::vector<std::pair<float3, float3>> kpt_correspondences, float3 rotation, int max_iterations=500) {
        Eigen::AngleAxisf rotation_x(rotation.x, Eigen::Vector3f::UnitZ());
        Eigen::AngleAxisf rotation_y(-rotation.y, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf rotation_z(rotation.z, Eigen::Vector3f::UnitX());

        std::vector<Eigen::Vector3f> ref_kpts;
        std::vector<Eigen::Vector3f> cmp_kpts;
        for (auto kpt_pair : kpt_correspondences) {
            ref_kpts.push_back(Eigen::Vector3f(kpt_pair.first.x, kpt_pair.first.y, kpt_pair.first.z));
            cmp_kpts.push_back(Eigen::Vector3f(kpt_pair.second.x, kpt_pair.second.y, kpt_pair.second.z));
        }

        auto translation_x = estimate_translation_single_axis(ref_kpts, cmp_kpts, rotation_x, rotation_y, rotation_z, max_iterations, EAxisX);
        auto translation_y = estimate_translation_single_axis(ref_kpts, cmp_kpts, rotation_x, rotation_y, rotation_z, max_iterations, EAxisY);
        auto translation_z = estimate_translation_single_axis(ref_kpts, cmp_kpts, rotation_x, rotation_y, rotation_z, max_iterations, EAxisZ);

        auto translation = Eigen::Translation3f(translation_x.x(),translation_y.y(), translation_z.z());
        return translation;
    }

    Eigen::Translation3f estimate_translation_single_axis(std::vector<Eigen::Vector3f> ref_kpts, std::vector<Eigen::Vector3f> cmp_kpts, Eigen::AngleAxisf rotation_x, Eigen::AngleAxisf rotation_y, Eigen::AngleAxisf rotation_z, int max_iterations, EAxis axis) {
        float magnitude = -1.0 * (max_iterations / 200.0);
        std::vector<Eigen::Translation3f> translations;
        for (int i = 0; i < max_iterations; i++) {
            translations.push_back(get_translation(magnitude, axis));
            magnitude += 0.01;
        }

        std::vector<Eigen::Vector3f> translated;
        std::vector<float> squares;
        for (auto translation : translations) {
            translated.clear();
            for (auto cmp_kpt : cmp_kpts)
                translated.push_back((translation * rotation_x * rotation_y * rotation_z) * cmp_kpt);

            float square_sum = 0;
            assert(translated.size() == ref_kpts.size());
            for (int i = 0; i < translated.size(); i++) {
                float diff = translated[i][axis] - ref_kpts[i][axis];
                square_sum += diff * diff;
            }
            squares.push_back(square_sum);
        }

        float least_square = squares[0];
        int least_idx = 0;
        for (int i = 1; i < squares.size(); i++)
            if (squares[i] < least_square) {
                least_square = squares[i];
                least_idx = i;
            }

        return translations[least_idx];
    }

    Eigen::Translation3f get_translation(float magnitude, EAxis axis) {
        switch (axis) {
            case EAxisX:
                return Eigen::Translation3f(magnitude, 0, 0);
            case EAxisY:
                return Eigen::Translation3f(0, magnitude, 0);
            case EAxisZ:
                return Eigen::Translation3f(0, 0, magnitude);
        }
    }
};

#endif //RS_PCL_TRANSLATION_ESTIMATOR_HPP
