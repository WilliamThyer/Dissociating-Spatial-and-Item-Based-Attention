"""A basic change detection experiment.

Author - William Thyer thyer@uchicago.edu

Adapted experiment code, originally from Colin Quirk's https://github.com/colinquirk/PsychopyChangeDetection 

If this file is run directly the defaults at the top of the page will be
used. To make simple changes, you can adjust any of these files. For more in depth changes you
will need to overwrite the methods yourself.

Note: this code relies on Colin Quirk's templateexperiments module. You can get it from
https://github.com/colinquirk/templateexperiments and either put it in the same folder as this
code or give the path to psychopy in the preferences.

Classes:
Cannonball01 -- The class that runs the experiment.
"""

import os
import sys
import errno

import json
import random
import copy

import numpy as np
import math

import psychopy.core
import psychopy.event
import psychopy.visual
import psychopy.parallel
import psychopy.tools.monitorunittools

import template 
import eyelinker

# Things you probably want to change
number_of_trials_per_block = 48 # must be divisible by 12
number_of_blocks = 24
percent_same = 0.5  # between 0 and 1
block_conditions = ['N1','H1','L1','2']
block_conditions_dict = {
    'N1': {'block_condition': 'N1', 'set_size': 1, 'num_digits': 1, 'num_letters': 0, 'num_placeholders': 0, 'code': 11, 'name': 'NO PLACEHOLDERS'},
    'H1': {'block_condition': 'H1', 'set_size': 1, 'num_digits': 1, 'num_letters': 0, 'num_placeholders': 1, 'code': 12, 'name': 'HASHTAG PLACEHOLDERS'},
    'L1': {'block_condition': 'L1', 'set_size': 1, 'num_digits': 1, 'num_letters': 1, 'num_placeholders': 1, 'code': 13, 'name': 'LETTER PLACEHOLDERS'},
    '2': {'block_condition': '2', 'set_size': 2, 'num_digits': 2, 'num_letters': 0, 'num_placeholders': 0, 'code': 20, 'name': '2 NUMBERS'},
    'L1F': {'block_condition': 'L1F', 'set_size': 1, 'num_digits': 1, 'num_letters': 1, 'num_placeholders': 1, 'code': 14, 'name': 'DIGIT PLACEHOLDERS'}
}

stim_size = 1.3  # visual degrees, used for X and Y

single_probe = True  # False to display all stimuli at test

distance_to_monitor = 90

instruct_text = [(
    'In this experiment you will be remembering digits.\n\n'
    'Each trial will start with a fixation cross. '
    'Do your best to keep your eyes on it at all times.\n'
    'An array of 1 or 2 digits will appear.\n'
    'Remember the digits and their locations as best you can.\n'
    'Ignore the hashtags and letters. You will not be tested on these.\n'
    'After a short delay, another digit will reappear.\n'
    'Use the UP and DOWN keys to select the digit from before.\n'
    'Use LEFT and RIGHT keys to cycle between different items.\n'
    'Press SPACE to confirm your selection.\n'
    'If you are not sure, just take your best guess.\n\n'
    'You will get breaks in between blocks.\n'
    "We'll start with some practice trials.\n\n"
    'Press the "S" key to start.'
)]

data_directory = os.path.join(
    '.', 'Data')

# Things you probably don't need to change, but can if you want to
exp_name = 'C01'

iti_time = .2 #this plus a 400:600 ms jittered iti
sample_time = 0.2
delay_time = .8

allowed_deg_from_fix = 4

# minimum euclidean distance between centers of stimuli in visual angle
# min_distance should be greater than stim_size
min_distance = 4
max_per_quad = 1  # int or None for totally random displays

letters = ['A','E','G','J','M','P','T','X']
digits = ['2','3','4','5','6','7','8','9']
stim_idx = [0,1,2,3,4,5,6,7]
num_pad_digits = ['num_2','num_3','num_4','num_5','num_6','num_7','num_8','num_9']

data_fields = [
    'Subject',
    'Block',
    'Trial',
    'Timestamp',
    'BlockCondition',
    'RT',
    'CRESP',
    'RESP',
    'ACC',
    'TargetLocation',
    'Locations',
    'Quadrants',
    'Digits',
    'Letters',
    'Stimuli',
    'NumPlaceholders'
]

gender_options = [
    'Male',
    'Female',
    'Other/Choose Not To Respond',
]

hispanic_options = [
    'Yes, Hispanic or Latino/a',
    'No, not Hispanic or Latino/a',
    'Choose Not To Respond',
]

race_options = [
    'American Indian or Alaskan Native',
    'Asian',
    'Pacific Islander',
    'Black or African American',
    'White / Caucasian',
    'More Than One Race',
    'Choose Not To Respond',
]

# Add additional questions here
questionaire_dict = {
    'Age': 0,
    'Gender': gender_options,
    'Hispanic:': hispanic_options,
    'Race': race_options,
}

# This is the logic that runs the experiment
# Change anything below this comment at your own risk
class Cannonball01(template.BaseExperiment):
    """The class that runs the  experiment.

    Parameters:
    allowed_deg_from_fix -- The maximum distance in visual degrees the stimuli can appear from
        fixation
    colors -- The list of colors (list of 3 values, -1 to 1) to be used in the experiment.
    orients -- The list of orientsations to be used in the experiment.
    stim_idx -- List of indices for colors and orientations.
    data_directory -- Where the data should be saved.
    delay_time -- The number of seconds between the stimuli display and test.
    instruct_text -- The text to be displayed to the participant at the beginning of the
        experiment.
    iti_time -- The number of seconds in between a response and the next trial.
    keys -- The keys to be used for making a response. First is used for 'same' and the second is
        used for 'different'
    max_per_quad -- The number of stimuli allowed in each quadrant. If None, displays are
        completely random.
    min_distance -- The minimum distance in visual degrees between stimuli.
    number_of_blocks -- The number of blocks in the experiment.
    number_of_trials_per_block -- The number of trials within each block.
    percent_same -- A float between 0 and 1 (inclusive) describing the likelihood of a trial being
        a "same" trial.
    questionaire_dict -- Questions to be included in the dialog.
    sample_time -- The number of seconds the stimuli are on the screen for.
    set_sizes -- A list of all the set sizes. An equal number of trials will be shown for each set
        size.
    single_probe -- If True, the test display will show only a single probe. If False, all the
        stimuli will be shown.
    stim_size -- The size of the stimuli in visual angle.

    Additional keyword arguments are sent to template.BaseExperiment().

    Methods:
    chdir -- Changes the directory to where the data will be saved.
    display_break -- Displays a screen during the break between blocks.
    display_fixation -- Displays a fixation cross.
    display_stimuli -- Displays the stimuli.
    display_test -- Displays the test array.
    generate_locations -- Helper function that generates locations for make_trial
    get_response -- Waits for a response from the participant.
    make_block -- Creates a block of trials to be run.
    make_trial -- Creates a single trial.
    run_trial -- Runs a single trial.
    run -- Runs the entire experiment.
    """

    def __init__(self, number_of_trials_per_block=number_of_trials_per_block,
                 number_of_blocks=number_of_blocks,
                 block_conditions=block_conditions, block_conditions_dict = block_conditions_dict,
                 stim_size=stim_size, letters=letters, num_pad_digits = num_pad_digits,
                 digits=digits, stim_idx = stim_idx, 
                 allowed_deg_from_fix=allowed_deg_from_fix,
                 min_distance=min_distance, max_per_quad=max_per_quad,
                 instruct_text=instruct_text, single_probe=single_probe,
                 iti_time=iti_time, sample_time=sample_time,
                 delay_time=delay_time, data_directory=data_directory,
                 questionaire_dict=questionaire_dict, **kwargs):

        self.number_of_trials_per_block = number_of_trials_per_block
        self.number_of_blocks = number_of_blocks
        self.block_conditions = block_conditions
        self.block_conditions_dict = block_conditions_dict
        self.stim_size = stim_size

        self.letters = letters
        self.lower_letters = [l.lower() for l in self.letters]
        self.digits = digits
        self.stim_idx = stim_idx
        self.num_pad_digits = num_pad_digits
        self.num_pad_dict = {
            'num_2':'2','num_3':'3','num_4':'4','num_5':'5','num_6':'6','num_7':'7','num_8':'8','num_9':'9'}\

        self.iti_time = iti_time
        self.sample_time = sample_time
        self.delay_time = delay_time

        self.allowed_deg_from_fix = allowed_deg_from_fix
        self.quad_dict = {0:(-1,1),1:(1,2),2:(1,-1),3:(-1,-1)}
        self.min_distance = min_distance
        self.max_per_quad = max_per_quad

        self.data_directory = data_directory
        self.instruct_text = instruct_text
        self.questionaire_dict = questionaire_dict

        self.single_probe = single_probe
        
        self.rej_counter = {'N1':0,'H1':0,'L1':0,'2':0,'L1F':0}
        self.rejection_tracker = np.zeros(5)

        super().__init__(**kwargs)

    def init_tracker(self):
        self.tracker = eyelinker.EyeLinker(
            self.experiment_window,
            self.experiment_name + '_' + self.experiment_info['Subject Number'] + '.edf',
            'BOTH')

        self.tracker.initialize_graphics()
        self.tracker.open_edf()
        self.tracker.initialize_tracker()
        self.tracker.send_tracking_settings()
    
    def show_eyetracking_instructions(self):
        self.tracker.display_eyetracking_instructions()
        self.tracker.setup_tracker()

    def start_eyetracking(self, block_num, trial_num):
        """Send block and trial status and start eyetracking recording

        Parameters:
        block_num-- Which block participant is in
        trial_num-- Which trial in block participant is in
        """
        status = f'Block {block_num+1}, Trial {trial_num+1}'
        self.tracker.send_status(status)

        self.tracker.start_recording()

    def stop_eyetracking(self):
        """Stop eyetracking recording
        """
        self.tracker.stop_recording()

    def realtime_eyetracking(self,wait_time,block_condition,sampling_rate=.01):
        """Collect real time eyetracking data over a period of time

        Returns eyetracking data

        Parameters:
        wait_time-- How long in ms to collect data for
        sampling_rate-- How many ms between each sample
        """
        start_time = psychopy.core.getTime()
        while psychopy.core.getTime() < start_time + wait_time:

            realtime_data = self.tracker.gaze_data

            reject,eyes = self.check_realtime_eyetracking(realtime_data)
#            reject=False
            if reject:
                self.rej_counter[block_condition] += 1
                
                print(f'# of rejected {block_condition} trials:{self.rej_counter[block_condition]}')
                    
                self.stop_eyetracking()
                self.display_eyemovement_feedback(eyes)
                return reject
            psychopy.core.wait(sampling_rate)
        
    def check_realtime_eyetracking(self,realtime_data):
        left_eye,right_eye = realtime_data
        if left_eye:
            lx,ly = left_eye
        if right_eye:
            rx,ry = right_eye        
        if (not left_eye) & (not right_eye):
            return False,None

        eyex = np.nanmean([lx,rx])
        eyey = np.nanmean([ly,ly])
        
        winx,winy = self.experiment_window.size/2
        
        eyex -= winx
        eyey -= winy
        eyes = np.array([eyex,eyey])

        limit_radius = psychopy.tools.monitorunittools.deg2pix(1.5,self.experiment_monitor)

        euclid_distance = np.linalg.norm(eyes-np.array([0,0])) 

        if euclid_distance > limit_radius:
            return True,(eyex,eyey)
        else:
            return False,None
            
    def display_eyemovement_feedback(self,eyes):

        psychopy.visual.TextStim(win=self.experiment_window,text='Eye Movement Detected',pos = [0,1], color = [1,-1,-1]).draw()
        psychopy.visual.TextStim(self.experiment_window, text='+', color=[-1, -1, -1]).draw()
        
        psychopy.visual.Circle(win=self.experiment_window,radius=5,pos=eyes,fillColor='red',units='pix').draw()
        
        self.experiment_window.flip()
        
        psychopy.core.wait(1.5)

    def handle_rejection(self,reject):
        self.rejection_tracker = np.roll(self.rejection_tracker,1)
        self.rejection_tracker[0] = reject
        
        if np.sum(self.rejection_tracker) == 5:
            self.rejection_tracker = np.zeros(5)
            self.display_text_screen(text='Rejected 5 in row\n\nContinue?',keyList = ['y'],bg_color=[0, 0, 255],text_color=[255,255,255])

    def kill_tracker(self):
        """Turns off eyetracker and transfers EDF file
        """
        self.tracker.set_offline_mode()
        self.tracker.close_edf()
        self.tracker.transfer_edf()
        self.tracker.close_connection()

    def setup_eeg(self):
        """ Connects the parallel port for EEG port code
        """
        try:
            self.port = psychopy.parallel.ParallelPort(address=53328)
        except:
            self.port = None
            print('No parallel port connected. Port codes will not send!')
        
    def send_synced_event(self, code, keyword = "SYNC"):
        """Send port code to EEG and eyetracking message for later synchronization

        Parameters:
        code-- Digits to send
        keyword-- Accompanying sync keyword (matters for later EEGLAB preprocessing)
        """

        message = keyword + ' ' + str(code)

        if self.port:
            self.port.setData(code)
            psychopy.core.wait(.005)
            self.port.setData(0)
            self.tracker.send_message(message)

    def chdir(self):
        """Changes the directory to where the data will be saved.
        """

        try:
            os.makedirs(self.data_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        os.chdir(self.data_directory)
    
    def make_block(self, block_num, number_of_trials_per_block = None, block_condition = None):

        if not number_of_trials_per_block:
            number_of_trials_per_block = self.number_of_trials_per_block
        if block_condition is None:
            block_condition = (int(self.experiment_info['Subject Number'])+block_num)%4
            block_dict = self.block_conditions_dict[block_conditions[block_condition]]
        else:
            block_dict = self.block_conditions_dict[block_condition]

        quads = np.empty((number_of_trials_per_block,2),dtype=np.int16)
        idx = 0
        while idx < quads.shape[0]:
            for i in [0,1,2,3]:
                for j in [0,1,2,3]:
                    if i != j:
                        quads[idx] = [i,j]
                        idx += 1
        
        block = []
        for itrial in range(number_of_trials_per_block):
            block.append(self.make_trial(block_dict,quads[itrial]))
        
        block = np.random.permutation(block)

        return block
        
    def make_trial(self, block_dict, quads):
        """Makes a single trial.

        Returns a dictionary of attributes about trial.
        """

        digit_idx = np.random.choice(self.stim_idx, size = block_dict['num_digits'], replace = False)
        letter_idx = np.random.choice(self.stim_idx, size = block_dict['num_letters'], replace = False)
        digits = [self.digits[d] for d in digit_idx]
        letters = [self.letters[l] for l in letter_idx]
        if block_dict['block_condition'] == 'H1':
            letters = ['#']

        locs = self.generate_locations(quads)

        quads = [int(a) for a in quads]

        if block_dict['block_condition'] is 'N1':
            locs = [locs[0]]
            quads = [quads[0]]
        
        if block_dict['block_condition'] is not 'L1F':
            cresp = digits
            stim = digits + letters
        else:
            cresp = letters
            stim = letters + digits

        if block_dict['block_condition'] == '2':
            target_locs = locs
        else:
            target_locs = locs[0]

        trial = {
            'code': int(block_dict['code']),
            'set_size': block_dict['set_size'],
            'block_condition': block_dict['block_condition'],
            'block_name': block_dict['name'],
            'num_digits': block_dict['num_digits'],
            'num_letters': block_dict['num_letters'],
            'num_placeholders': block_dict['num_placeholders'],
            'locations': locs,
            'target_locations': target_locs,
            'quads': quads,
            'stim': stim,
            'digits': digits,
            'letters': letters,
            'cresp': cresp
        }

        return trial

    def _too_close(self, attempt, locs):
        """Checks that an attempted location is valid.

        This method is used by generate_locations to ensure the min_distance condition is followed.

        Parameters:
        attempt -- A list of two values (x,y) in visual angle.
        locs -- A list of previous successful attempts to be checked.
        """
        
        # Too close to center
        if np.linalg.norm(np.array(attempt)) < self.min_distance/1.5:
            return True  
            
        for loc in locs:
            # Too close to another square
            if np.linalg.norm(np.array(attempt) - np.array(loc)) < self.min_distance:
                return True  
            # Too close vertically to each other
            if abs(attempt[0] - loc[0]) < self.min_distance/2:
                return True

        return False

    def generate_locations(self, quads):
        """Creates the locations for a trial. A helper function for self.make_trial.

        Returns a list of acceptable locations.

        Parameters:
        set_size -- The number of stimuli for this trial.
        """

        locs = []
        counter = 0
        iquad = 0

        while len(locs) < 2:
            counter += 1
            if counter > 1000000:
                raise ValueError('Timeout -- Cannot generate locations with given values.')

            # generate x,y within min and max degree from fix
            attempt = [random.uniform(0, self.allowed_deg_from_fix) for _ in range(2)]
            # multiply x,y by -1 or 1 to put x,y into correct quad for trial
            attempt = [round(a*b,3) for a,b in zip(attempt,self.quad_dict[quads[iquad]])]

            if self._too_close(attempt, locs):
                continue
            else:
                locs.append(attempt)
                iquad += 1

        return locs

    def display_start_block_screen(self,block_name):

        self.display_text_screen(text=f'This is a {block_name} block.\nPress space to begin.',keyList=['space'])

    def display_fixation(self, wait_time = None, text = None, keyList = None, realtime_eyetracking = False, trial = None):
        """Displays a fixation cross. A helper function for self.run_trial.

        Parameters:
        wait_time -- The amount of time the fixation should be displayed for.
        text -- Str that displays above fixation cross. 
        keyList -- If keyList is given, will wait until key press
        trial -- Trial object needed for realtime eyetracking functionality.
        real_time_eyetracking -- Bool for if you want to do realtime eyetracking or not
        """
        
        if text:
            psychopy.visual.TextStim(win=self.experiment_window,text=text,pos = [0,1], color = [1,-1,-1]).draw()

        psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1]).draw()
        
        self.experiment_window.flip()

        if realtime_eyetracking:
            reject = self.realtime_eyetracking(wait_time=wait_time,block_condition=trial['block_condition'])
            return reject    
        else:
            if keyList:
                resp = psychopy.event.waitKeys(maxWait=wait_time,keyList=keyList)
                if resp == ['p']:
                    self.display_text_screen(text='Paused',keyList = ['space'])
                    self.display_fixation(wait_time=1)
                elif resp == ['o']:
                    self.tracker.calibrate()
                    self.display_fixation(wait_time=1)
                elif resp == ['escape']:
                    resp = self.display_text_screen(text='Are you sure you want to exit?',keyList = ['y','n'])
                    if resp == ['y']:
                        self.tracker.transfer_edf()
                        self.quit_experiment()
                    else:
                        self.display_fixation(wait_time=1)
            else:
                psychopy.core.wait(wait_time)
                
    def draw_stim(self,text,pos):
        text_stim = psychopy.visual.TextStim(
            self.experiment_window, text=text, color='black', pos=pos, height=2)
        
        text_stim.draw()
    
    def draw_trak(self,x=930, y=510):
        trak = psychopy.visual.Circle(
            self.experiment_window, lineColor=None, fillColor = [1,1,1], 
            fillColorSpace='rgb', radius=20, pos = [x,y], units='pix'
        )
        
        trak.draw()

    def display_stimuli(self, trial, realtime_eyetracking=False):
        """Displays the stimuli. A helper function for self.run_trial.

        Parameters:
        locations -- A list of locations (list of x and y value) describing where the stimuli
            should be displayed.
        colors -- A list of colors describing what should be drawn at each coordinate.
        """

        psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1]).draw()
        
        for loc,stim in zip(trial['locations'],trial['stim']):
            self.draw_stim(stim,loc)

        self.draw_trak()
        self.send_synced_event(trial['code'])
        self.experiment_window.flip()

        if realtime_eyetracking:  
            reject = self.realtime_eyetracking(wait_time=self.sample_time,block_condition=trial['block_condition'])
            return reject
        else:
            psychopy.core.wait(self.sample_time)

    def get_responses(self, trial):
        
        self.send_synced_event(3) #success code means no eyetracking rejection
        
        fix = psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1])

        rect = psychopy.visual.Rect(
            win=self.experiment_window, width = 2, height = 2.5, lineColor='black', fillColor=None,
            fillColorSpace='rgb', lineWidth=10)
        
        responses_idx = list(np.random.choice(self.stim_idx, size = trial['set_size'], replace = False))
        responses = ['']*trial['set_size']

        rt_timer = psychopy.core.MonotonicClock()

        item_idx = 0
        cont = False
        while cont is False:
        
            if trial['block_condition'] is not 'L1F':
                for i,idx in enumerate(responses_idx):
                    responses[i] = self.digits[idx]
            else:
                for i,idx in enumerate(responses_idx):
                    responses[i] = self.letters[idx]

            for i in range(trial['set_size']):
            
                if i == item_idx:
                    rect.lineColor = 'red'
                else:
                    rect.lineColor = 'black'
                rect.pos = trial['locations'][i]
                rect.draw()

                self.draw_stim(responses[i],trial['locations'][i])
                fix.draw()

            self.experiment_window.flip()
            
            resp = psychopy.event.waitKeys()

            if (resp[0] == 'space') & ('' not in responses):
                rt = rt_timer.getTime()*1000
                cont = True

            if (resp[0] == 'left' or resp[0] == 'right') & (trial['block_condition'] == '2'):
                if item_idx == 0:
                    item_idx = 1
                else:
                    item_idx = 0
            
            if resp[0] == 'up':
                if trial['block_condition'] == 'L1F':
                    responses_idx[item_idx] -= 1
                else:    
                    responses_idx[item_idx] += 1
            if resp[0] == 'down':
                if trial['block_condition'] == 'L1F':
                    responses_idx[item_idx] += 1
                else:    
                    responses_idx[item_idx] -= 1
            
            if responses_idx[item_idx] < 0:
                    responses_idx[item_idx] = 7
            if responses_idx[item_idx] > 7:
                    responses_idx[item_idx] = 0    

        return responses, rt

    def send_data(self, data):
        """Updates the experiment data with the information from the last trial.

        This function is seperated from run_trial to allow additional information to be added
        afterwards.

        Parameters:
        data -- A dict where keys exist in data_fields and values are to be saved.
        """
        self.update_experiment_data([data])

    def run_trial(self, trial, block_num, trial_num, realtime_eyetracking=False):
        """Runs a single trial.

        Returns the data from the trial after getting a participant response.

        Parameters:
        trial -- The dictionary of information about a trial.
        block_num -- The number of the block in the experiment.
        trial_num -- The number of the trial within a block.
        """
        self.display_fixation(wait_time=np.random.randint(400,601)/1000,trial=trial,keyList=['p','escape','o'])
        self.start_eyetracking(block_num = block_num, trial_num = trial_num)
        
        self.send_synced_event(1)
        reject = self.display_fixation(wait_time=self.iti_time,trial=trial, realtime_eyetracking=realtime_eyetracking)
        if reject:
            self.handle_rejection(1)
            return None

        reject = self.display_stimuli(
            trial=trial, realtime_eyetracking=realtime_eyetracking)
        if reject:
            self.handle_rejection(1)
            return None

        self.send_synced_event(2)
        reject = self.display_fixation(self.delay_time, trial=trial, realtime_eyetracking=realtime_eyetracking)
        if reject:
            self.handle_rejection(1)
            return None

        resp, rt = self.get_responses(trial)

        self.send_synced_event(4)
        self.stop_eyetracking()
        self.handle_rejection(0)
        
        acc = 1 if resp == trial['cresp'] else 0

        data = {
            'Subject': self.experiment_info['Subject Number'],
            'Block': block_num,
            'Trial': trial_num,
            'Timestamp': psychopy.core.getAbsTime(),
            'BlockCondition': trial['block_condition'],
            'SetSize': json.dumps(trial['set_size']),
            'RT': rt,
            'CRESP': trial['cresp'],
            'RESP': resp,
            'ACC': acc,
            'Locations': json.dumps(trial['locations']),
            'TargetLocation': json.dumps(trial['target_locations']),
            'Quadrants': json.dumps(trial['quads']),
            'Digits': trial['digits'],
            'Letters': trial['letters'],
            'Stimuli': trial['stim'],
            'NumPlaceholders': json.dumps(trial['num_placeholders'])
        }

        print(f'{block_num+1}, {trial_num+1}')
        print(f'Acc:{acc}')
        return data

    def run_makeup_block(self,block_condition,block_num):

            num_makeup_trials = copy.copy(self.rej_counter[block_condition])
            num_makeup_trials = math.ceil(num_makeup_trials / 12) * 12
            self.rej_counter[block_condition] = 0
            
            self.rejection_tracker = np.zeros(5)
            block = self.make_block(block_num=block_num,
                                    number_of_trials_per_block=num_makeup_trials,
                                    block_condition=block_condition)
            
            acc = []
            self.tracker.calibrate()
            for trial_num, trial in enumerate(block):
                if trial_num == 0:
                    self.display_start_block_screen(trial['block_name'])
                if trial_num % 5 == 0:
                    self.tracker.drift_correct()
                    
                data = self.run_trial(trial, block_num, trial_num, realtime_eyetracking=True)
                if data:
                    self.send_data(data)
                    acc.append(data['ACC'])

            self.save_data_to_csv()
            self.display_text_screen(
                text = f'Block Accuracy: {round(100*np.nanmean(acc))}\n\n\n\nPress space to continue.',keyList=['space'])

    def run(self):
        """Runs the entire experiment.

        This function takes a number of hooks that allow you to alter behavior of the experiment
        without having to completely rewrite the run function. While large changes will still
        require you to create a subclass, small changes like adding a practice block or
        performance feedback screen can be implimented using these hooks. All hooks take in the
        experiment object as the first argument. See below for other parameters sent to hooks.

        Parameters:
        setup_hook -- takes self, executed once the window is open.
        before_first_trial_hook -- takes self, executed after instructions are displayed.
        pre_block_hook -- takes self, block list, and block num
            Executed immediately before block start.
            Can optionally return an altered block list.
        pre_trial_hook -- takes self, trial dict, block num, and trial num
            Executed immediately before trial start.
            Can optionally return an altered trial dict.
        post_trial_hook -- takes self and the trial data, executed immediately after trial end.
            Can optionally return altered trial data to be stored.
        post_block_hook -- takes self, executed at end of block before break screen (including
            last block).
        end_experiment_hook -- takes self, executed immediately before end experiment screen.
        """

        """
        Setup and Instructions
        """
        self.chdir()

        ok = self.get_experiment_info_from_dialog(self.questionaire_dict)

        if not ok:
            print('Experiment has been terminated.')
            sys.exit(1)

        self.save_experiment_info()
        self.open_csv_data_file(data_filename = self.experiment_name + '_' + self.experiment_info['Subject Number'])
        self.open_window(screen=0)
        self.display_text_screen('Loading...', wait_for_input=False)

        self.init_tracker()

        for instruction in self.instruct_text:
            self.display_text_screen(text=instruction, keyList=['space'])
        
        self.show_eyetracking_instructions()

        """
        Practice
        """
        block_num = 0
        self.port = None
        prac = self.display_text_screen(text = f'Practice block?', keyList=['y','n'])
        
        while prac == ['y']: 
            
            block = self.make_block(block_num,number_of_trials_per_block=12)
            acc = []
            
            for trial_num, trial in enumerate(block):
                if trial_num == 0:
                    self.display_start_block_screen(trial['block_name'])
                
                data = self.run_trial(trial,block_num,trial_num)      
                acc.append(data['ACC'])

            block_num += 1
            
            self.display_text_screen(
                text = f'Block Accuracy: {round(100*np.nanmean(acc))}\n\n\n\nPress space to continue.',keyList=['space'])
        
            prac = self.display_text_screen(text = f'Practice block?', keyList=['y','n'])
            
        """
        Experiment
        """
        # N1, H1, L1, & 2 Blocks
        self.setup_eeg()
        for block_num in range(self.number_of_blocks):
            block = self.make_block(block_num)
            acc = []
            
            self.tracker.calibrate()
            self.rejection_tracker = np.zeros(5)
            for trial_num, trial in enumerate(block):
                if trial_num == 0:
                    self.display_start_block_screen(trial['block_name'])
                if trial_num % 5 == 0:
                    self.tracker.drift_correct()

                data = self.run_trial(trial, block_num, trial_num, realtime_eyetracking=True)
                if data:
                    self.send_data(data)
                    acc.append(data['ACC'])

            self.save_data_to_csv()

            self.display_text_screen(
                text = f'Block Accuracy: {round(100*np.nanmean(acc))}\n\n\n\nPress space to continue.',keyList=['space'])

        """
        Makeup Blocks
        """        

        for block_con in self.block_conditions:
            while self.rej_counter[block_con] > 35:
                block_num += 1
                self.run_makeup_block(block_condition=block_con,block_num=block_num)
                
                
        """
        L1F Blocks
        """
        for _ in range(2):
            block_num += 1
            block = self.make_block(block_num,block_condition='L1F')
            acc = []
            
            self.tracker.calibrate()
            self.rejection_tracker = np.zeros(5)
            for trial_num, trial in enumerate(block):
                if trial_num == 0:
                    self.display_start_block_screen(trial['block_name'])
                if trial_num % 5 == 0:
                    self.tracker.drift_correct()

                data = self.run_trial(trial, block_num, trial_num, realtime_eyetracking=True)
                if data:
                    self.send_data(data)
                    acc.append(data['ACC'])

            self.save_data_to_csv()

            self.display_text_screen(
                text = f'Block Accuracy: {round(100*np.nanmean(acc))}\n\n\n\nPress space to continue.', keyList=['space'])

        while self.rej_counter['L1F'] > 15:
            block_num += 1
            self.run_makeup_block(block_condition=block_con,block_num=block_num)

        """
        End of Experiment
        """
        self.display_text_screen(
            'The experiment is now over, please get your experimenter.',
            bg_color=[0, 0, 255], text_color=[255, 255, 255])
        
        self.tracker.transfer_edf()
        self.quit_experiment()

# If you call this script directly, the task will run with your defaults
if __name__ == '__main__':
    exp = Cannonball01(
        # BaseExperiment parameters
        experiment_name=exp_name,
        data_fields=data_fields,
        monitor_distance=distance_to_monitor,
        # Custom parameters go here
    )

    try:
        exp.run()
    except Exception as e:
        exp.kill_tracker()
        raise e
