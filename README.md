# Dissociating Spatial and Item-Based Attention
Experiment, preprocessing, and analysis code for Ch 3 of my dissertation. "Cannonball" and "Discus" studies. Uses https://github.com/colinquirk/templateexperiments to build experiments. Includes realtime eyetracking rejection and synchronous EEG port codes.

# Experiments

## Cannonball 

Experiment 1. Digit and letter change detection with hashtag (#) distractors.

## Discus 

Experiment 2. Color change detection. Target squares and ignore rectangles. 

# Artifact rejection

## eegreject.m

Main script that handles alignment, epoching, other preprocessing, and artifact rejection.

## align_channels.m

Realign eyetracking, EOG, and stimtrak to make channels more visible during inspection of EEG for manual rejection.

## data_from_checked_eeg.m

Pull code from matlab and save it in .mat file for later analysis in Python.
