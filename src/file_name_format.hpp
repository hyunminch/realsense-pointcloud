//
// Created by eundolee on 19. 12. 19..
//

#ifndef RS_PCL_FILE_NAME_FORMAT_HPP
#define RS_PCL_FILE_NAME_FORMAT_HPP

//enum ECorKpt {
//    EIn=0,
//    EOut
//};

const std::string PREFIX_DIR = "dataset/";
const std::string SUFFIX_PCD = ".pcd";
const std::string SUFFIX_TXT = ".txt";
const std::string SUFFIX_JPG = ".jpg";
const std::string DELIMITER = "-";

const std::string RAW = "raw";
const std::string MATCHES = "matches";
const std::string COR_KPTS = "cor-kpts";
const std::string IN = "in";
const std::string OUT = "out";
const std::string TRANSMAT = "transmat";
const std::string TO = "to";
const std::string NO_BLUR = "no-blur";
const std::string GUESS = "guess";
const std::string RESULT = "result";
const std::string UPTO = "upto";
const std::string RESULT_UPTO = RESULT + DELIMITER + UPTO;

std::string fn_range(int from, int to) {
    std::ostringstream o;
    o << from << DELIMITER << TO << DELIMITER << to;
    return o.str();
}

std::string fn_raw(const std::string prefix, int idx) {
    std::ostringstream o;
    o << PREFIX_DIR << prefix << DELIMITER << RAW << DELIMITER << idx << SUFFIX_PCD;
    return o.str();
}

std::string fn_matches(const std::string prefix, int idx) {
    std::ostringstream o;
    o << PREFIX_DIR << prefix << DELIMITER << MATCHES << DELIMITER << idx << SUFFIX_JPG;
    return o.str();
}

std::string fn_cor_kpts(const std::string prefix, int from, int to) {
    std::ostringstream o;
    o << PREFIX_DIR << prefix << DELIMITER << COR_KPTS << DELIMITER << fn_range(from, to) << SUFFIX_PCD;
    return o.str();
}

std::string fn_transmat(const std::string prefix, int from, int to) {
    std::ostringstream o;
    o << PREFIX_DIR << prefix << DELIMITER << TRANSMAT << DELIMITER << fn_range(from, to) << SUFFIX_TXT;
    return o.str();
}

std::string fn_no_blur(const std::string prefix, int idx) {
    std::ostringstream o;
    o << PREFIX_DIR << prefix << DELIMITER << NO_BLUR << DELIMITER << idx << SUFFIX_PCD;
    return o.str();
}

std::string fn_guess(const std::string prefix, int from, int to) {
    std::ostringstream o;
    o << PREFIX_DIR << prefix << DELIMITER << GUESS << DELIMITER << fn_range(from, to) << SUFFIX_TXT;
    return o.str();
}

std::string fn_result_upto(const std::string prefix, int idx) {
    std::ostringstream o;
    o << PREFIX_DIR << prefix << DELIMITER << RESULT_UPTO << DELIMITER << idx << SUFFIX_PCD;
    return o.str();
}

std::string fn_result(const std::string prefix) {
    std::ostringstream o;
    o << PREFIX_DIR << prefix << DELIMITER << RESULT << SUFFIX_PCD;
    return o.str();
}

#endif //RS_PCL_FILE_NAME_FORMAT_HPP
