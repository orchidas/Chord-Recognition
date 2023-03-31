import numpy as np
import os, sys, getopt
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import json
from chromagram import compute_chroma
import hmm as hmm


def get_templates(chords):
    """read from JSON file to get chord templates"""
    with open("data/chord_templates.json", "r") as fp:
        templates_json = json.load(fp)
    templates = []

    for chord in chords:
        if chord == "N":
            continue
        templates.append(templates_json[chord])

    return templates


def get_nested_circle_of_fifths():
    chords = [
        "N",
        "G",
        "G#",
        "A",
        "A#",
        "B",
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "Gm",
        "G#m",
        "Am",
        "A#m",
        "Bm",
        "Cm",
        "C#m",
        "Dm",
        "D#m",
        "Em",
        "Fm",
        "F#m",
    ]
    nested_cof = [
        "G",
        "Bm",
        "D",
        "F#m",
        "A",
        "C#m",
        "E",
        "G#m",
        "B",
        "D#m",
        "F#",
        "A#m",
        "C#",
        "Fm",
        "G#",
        "Cm",
        "D#",
        "Gm",
        "A#",
        "Dm",
        "F",
        "Am",
        "C",
        "Em",
    ]
    return chords, nested_cof


def find_chords(
    x: np.ndarray,
    fs: int,
    templates: list,
    chords: list,
    nested_cof: list = None,
    method: str = None,
    plot: bool = False,
):
    """
    Given a mono audio signal x, and its sampling frequency, fs,
    find chords in it using 'method'
    Args:
        x : mono audio signal
        fs : sampling frequency (Hz)
        templates: dictionary of chord templates
        chords: list of chords to search over
        nested_cof: nested circle of fifth chords
        method: template matching or HMM
        plot: if results should be plotted
    """

    # framing audio, window length = 8192, hop size = 1024 and computing PCP
    nfft = 8192
    hop_size = 1024
    nFrames = int(np.round(len(x) / (nfft - hop_size)))
    # zero padding to make signal length long enough to have nFrames
    x = np.append(x, np.zeros(nfft))
    xFrame = np.empty((nfft, nFrames))
    start = 0
    num_chords = len(templates)
    chroma = np.empty((num_chords // 2, nFrames))
    id_chord = np.zeros(nFrames, dtype="int32")
    timestamp = np.zeros(nFrames)
    max_cor = np.zeros(nFrames)

    # step 1. compute chromagram
    for n in range(nFrames):
        xFrame[:, n] = x[start : start + nfft]
        start = start + nfft - hop_size
        timestamp[n] = n * (nfft - hop_size) / fs
        chroma[:, n] = compute_chroma(xFrame[:, n], fs)

    if method == "match_template":
        # correlate 12D chroma vector with each of
        # 24 major and minor chords
        for n in range(nFrames):
            cor_vec = np.zeros(num_chords)
            for ni in range(num_chords):
                cor_vec[ni] = np.correlate(chroma[:, n], np.array(templates[ni]))
            max_cor[n] = np.max(cor_vec)
            id_chord[n] = np.argmax(cor_vec) + 1

        # if max_cor[n] < threshold, then no chord is played
        # might need to change threshold value
        id_chord[np.where(max_cor < 0.8 * np.max(max_cor))] = 0
        final_chords = [chords[cid] for cid in id_chord]

    elif method == "hmm":
        # get max probability path from Viterbi algorithm
        (PI, A, B) = hmm.initialize(chroma, templates, chords, nested_cof)
        (path, states) = hmm.viterbi(PI, A, B)

        # normalize path
        for i in range(nFrames):
            path[:, i] /= sum(path[:, i])

        # choose most likely chord - with max value in 'path'
        final_chords = []
        indices = np.argmax(path, axis=0)
        final_states = np.zeros(nFrames)

        # find no chord zone
        set_zero = np.where(np.max(path, axis=0) < 0.3 * np.max(path))[0]
        if np.size(set_zero) > 0:
            indices[set_zero] = -1

        # identify chords
        for i in range(nFrames):
            if indices[i] == -1:
                final_chords.append("NC")
            else:
                final_states[i] = states[indices[i], i]
                final_chords.append(chords[int(final_states[i])])

    if plot:
        plt.figure()
        plt.yticks(np.arange(num_chords + 1), chords)
        plt.plot(timestamp, id_chord)
        plt.xlabel("Time in seconds")
        plt.ylabel("Chords")
        plt.title("Identified chords")
        plt.grid(True)
        plt.show()

    return timestamp, final_chords


def main(argv):
    input_file = ""
    method = ""
    plot = False
    has_method = False
    try:
        opts, args = getopt.getopt(argv, "hi:m:p:", ["ifile=", "method=", "plot="])
    except getopt.GetoptError:
        print("main.py -i <inputfile> -m <method>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("main.py -i <input_file> -m <method> -p <plot>")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-m", "--method"):
            method = arg
            has_method = True
        elif opt in ("-p", "--plot"):
            plot = True
    if not has_method:
        method = "match_template"

    print("Input file is ", input_file)
    print("Method is ", method)
    directory = os.getcwd() + "/data/test_chords/"
    # read the input file
    (fs, s) = read(directory + input_file)
    # convert to mono if file is stereo
    x = s[:, 0] if len(s.shape) else s

    # get chords and circle of fifths
    chords, nested_cof = get_nested_circle_of_fifths()
    # get chord templates
    templates = get_templates(chords)

    # find the chords
    if method == "match_template":
        timestamp, final_chords = find_chords(
            x, fs, templates=templates, chords=chords, method=method, plot=plot
        )
    else:
        timestamp, final_chords = find_chords(
            x,
            fs,
            templates=templates,
            chords=chords[1:],
            nested_cof=nested_cof,
            method=method,
            plot=plot,
        )

    # print chords with timestamps
    print("Time (s)", "Chord")
    for n in range(len(timestamp)):
        print("%.3f" % timestamp[n], final_chords[n])


if __name__ == "__main__":
    main(sys.argv[1:])
