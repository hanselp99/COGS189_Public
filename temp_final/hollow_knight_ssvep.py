"""
Hollow Knight SSVEP BCI Controller
====================================
2x4 grid of flickering stimuli mapped to Hollow Knight controls.
Runs on a second monitor as a separate window alongside the game.

Grid layout:
  (0,0) Left+Jump   | (0,1) Jump    | (0,2) Dash   | (0,3) Right+Jump
  (1,0) Left        | (1,1) Attack  | (1,2) Focus  | (1,3) Right

Key mappings:
  Left+Jump  -> left + z  (held until new action)
  Jump       -> z         (tap)
  Dash       -> c         (tap)
  Right+Jump -> right + z (held until new action)
  Left       -> left      (held until new action)
  Attack     -> x         (tap)
  Focus      -> a         (tap)
  Right      -> right     (held until new action)

Dependencies:
    pip install psychopy brainflow mne pyautogui numpy scipy
"""

from psychopy import visual, core
from psychopy.hardware import keyboard as kb
import numpy as np
from scipy import signal
import os, pickle, time, random
from threading import Thread, Event
from queue import Queue
import pyautogui

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
cyton_in        = True
width           = 800       # second monitor window width
height          = 400       # second monitor window height
refresh_rate    = 60.0
stim_duration   = 1.2       # seconds per trial
n_per_class     = 2         # trials per class for calibration
subject         = 1
session         = 1
run             = 1
calibration_mode = True    # True = collect data, False = live play

save_dir        = f'data/hollow_knight_ssvep/sub-{subject:02d}/ses-{session:02d}/'
model_file_path = 'cache/FBTRCA_model_hk.pkl'

os.makedirs(save_dir, exist_ok=True)

save_file_eeg        = save_dir + f'eeg_{n_per_class}-per-class_run-{run}.npy'
save_file_aux        = save_dir + f'aux_{n_per_class}-per-class_run-{run}.npy'
save_file_eeg_trials = save_dir + f'eeg-trials_{n_per_class}-per-class_run-{run}.npy'
save_file_aux_trials = save_dir + f'aux-trials_{n_per_class}-per-class_run-{run}.npy'

# ─────────────────────────────────────────────
# ACTION DEFINITIONS
# ─────────────────────────────────────────────
# Each action: (label, [keys], is_held)
# is_held = True  -> hold until next trial predicts something different
# is_held = False -> single tap on prediction

ACTIONS = [
    # Row 0 (top)
    {'label': 'Left\nJump',  'keys': ['left', 'z'],     'held': True},   # 0
    {'label': 'Jump',        'keys': ['z'],              'held': False},  # 1
    {'label': 'Dash',        'keys': ['c'],              'held': False},  # 2
    {'label': 'Right\nJump', 'keys': ['right', 'z'],    'held': True},   # 3
    # Row 1 (bottom)
    {'label': 'Left',        'keys': ['left'],           'held': True},   # 4
    {'label': 'Attack',      'keys': ['x'],              'held': False},  # 5
    {'label': 'Focus',       'keys': ['a'],              'held': False},  # 6
    {'label': 'Right',       'keys': ['right'],          'held': True},   # 7
]

N_CLASSES = 8   # 2 rows × 4 cols

# ─────────────────────────────────────────────
# SSVEP STIMULUS CLASSES
# 8 unique frequency+phase combos for 8 targets
# ─────────────────────────────────────────────
stimulus_classes = [
    (8,  0.0),   # Left+Jump
    (9,  0.0),   # Jump
    (10, 0.0),   # Dash
    (11, 0.0),   # Right+Jump
    (8,  0.5),   # Left
    (9,  0.5),   # Attack
    (10, 0.5),   # Focus
    (11, 0.5),   # Right
]

num_frames     = int(np.round(stim_duration * refresh_rate))
frame_indices  = np.arange(num_frames)

# Pre-compute per-frame colour values for each stimulus
stimulus_frames = np.zeros((num_frames, N_CLASSES))
for i_class, (freq, phase) in enumerate(stimulus_classes):
    phase_adj = phase + 1e-5
    stimulus_frames[:, i_class] = signal.square(
        2 * np.pi * freq * (frame_indices / refresh_rate) + phase_adj * np.pi
    )

# ─────────────────────────────────────────────
# GRID LAYOUT  (2 rows × 4 cols)
# ─────────────────────────────────────────────
N_ROWS, N_COLS = 2, 4

def get_cell_positions(cell_w, cell_h, win_w, win_h):
    """Return normalised (x, y) centre positions for each cell, row-major."""
    positions = []
    for row in range(N_ROWS):
        for col in range(N_COLS):
            x = -1 + cell_w / 2 + col * cell_w
            y =  1 - cell_h / 2 - row * cell_h
            positions.append((x, y))
    return positions

# Cell size in normalised units
cell_w = 2.0 / N_COLS   # 0.5
cell_h = 2.0 / N_ROWS   # 1.0
cell_positions = get_cell_positions(cell_w, cell_h, width, height)

# ─────────────────────────────────────────────
# PSYCHOPY WINDOW
# ─────────────────────────────────────────────
# Place window on second monitor by offsetting by primary monitor width.
# Adjust `screen=1` or the winPos if your setup differs.
window = visual.Window(
    size=[width, height],
    fullscr=False,
    allowGUI=True,          # True so the window can be moved/resized freely
    color=[-0.5, -0.5, -0.5],
    units='norm',
)

keyboard = kb.Keyboard()

# ─────────────────────────────────────────────
# VISUAL ELEMENTS
# ─────────────────────────────────────────────

def create_stimulus_rects():
    """One rectangle per action cell."""
    rects = []
    for i, (x, y) in enumerate(cell_positions):
        rect = visual.Rect(
            win=window,
            width=cell_w * 0.92,
            height=cell_h * 0.92,
            pos=(x, y),
            fillColor=[-1, -1, -1],
            lineColor=[0.3, 0.3, 0.3],
            lineWidth=2,
            units='norm',
        )
        rects.append(rect)
    return rects


def create_label_texts():
    """Action label over each cell."""
    texts = []
    for i, action in enumerate(ACTIONS):
        x, y = cell_positions[i]
        txt = visual.TextStim(
            win=window,
            text=action['label'],
            pos=(x, y),
            color=[1, 1, 1],
            height=0.12,
            units='norm',
            alignText='center',
        )
        texts.append(txt)
    return texts


def create_photosensor_dot():
    return visual.Rect(
        win=window,
        units='norm',
        width=0.06,
        height=0.12,
        fillColor='white',
        lineWidth=0,
        pos=(1 - 0.03, -1 + 0.06),
    )


def create_highlight_rect(pos):
    """Highlight box drawn around the currently predicted cell."""
    return visual.Rect(
        win=window,
        width=cell_w * 0.96,
        height=cell_h * 0.96,
        pos=pos,
        fillColor=None,
        lineColor=[0.0, 1.0, 0.0],
        lineWidth=4,
        units='norm',
    )


stim_rects      = create_stimulus_rects()
label_texts     = create_label_texts()
photosensor_dot = create_photosensor_dot()

# ─────────────────────────────────────────────
# OPENBCI CYTON SETUP
# ─────────────────────────────────────────────
if cyton_in:
    import glob, sys, serial
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    from serial import Serial

    sampling_rate  = 250
    CYTON_BOARD_ID = 0
    BAUD_RATE      = 115200
    ANALOGUE_MODE  = '/2'

    def find_openbci_port():
        if sys.platform.startswith('win'):
            ports = ['COM%s' % (i + 1) for i in range(256)]
        elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
            ports = glob.glob('/dev/ttyUSB*')
        elif sys.platform.startswith('darwin'):
            ports = glob.glob('/dev/cu.usbserial*')
        else:
            raise EnvironmentError('Unsupported OS for port detection')
        for port in ports:
            try:
                s = Serial(port=port, baudrate=BAUD_RATE, timeout=None)
                s.write(b'v')
                time.sleep(2)
                if s.inWaiting():
                    line = ''
                    while '$$$' not in line:
                        line += s.read().decode('utf-8', errors='replace')
                    if 'OpenBCI' in line:
                        s.close()
                        return port
                s.close()
            except (OSError, serial.SerialException):
                pass
        raise OSError('Cannot find OpenBCI port.')

    params = BrainFlowInputParams()
    params.serial_port = find_openbci_port()
    board = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    board.config_board('/0')
    board.config_board('//')
    board.config_board(ANALOGUE_MODE)
    board.start_stream(45000)
    stop_event = Event()

    eeg_queue = Queue()

    def stream_data(queue, stop):
        while not stop.is_set():
            data = board.get_board_data()
            ts  = data[board.get_timestamp_channel(CYTON_BOARD_ID)]
            eeg = data[board.get_eeg_channels(CYTON_BOARD_ID)]
            aux = data[board.get_analog_channels(CYTON_BOARD_ID)]
            if len(ts) > 0:
                queue.put((eeg, aux, ts))
            time.sleep(0.1)

    cyton_thread        = Thread(target=stream_data, args=(eeg_queue, stop_event))
    cyton_thread.daemon = True
    cyton_thread.start()

# ─────────────────────────────────────────────
# LOAD MODEL (if exists)
# ─────────────────────────────────────────────
model = None
if os.path.exists(model_file_path):
    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)
    print('Model loaded from', model_file_path)
else:
    print('No model found at', model_file_path, '— predictions will be random placeholders.')

# ─────────────────────────────────────────────
# KEY PRESS HELPER
# ─────────────────────────────────────────────
_currently_held_keys = []

def release_held_keys():
    global _currently_held_keys
    for k in _currently_held_keys:
        pyautogui.keyUp(k)
    _currently_held_keys = []


def execute_action(action_idx, previous_idx):
    """
    Execute the Hollow Knight key action for the predicted class.
    - Held actions: release previous keys, press & hold new ones.
    - Tap actions:  release any held keys first, then tap.
    """
    global _currently_held_keys
    action = ACTIONS[action_idx]

    if action['held']:
        # Only change keys if prediction changed
        if action_idx != previous_idx:
            release_held_keys()
            for k in action['keys']:
                pyautogui.keyDown(k)
            _currently_held_keys = list(action['keys'])
    else:
        # Tap: release any currently held movement keys first
        release_held_keys()
        # Press all combo keys simultaneously
        for k in action['keys']:
            pyautogui.keyDown(k)
        time.sleep(0.05)
        for k in action['keys']:
            pyautogui.keyUp(k)


# ─────────────────────────────────────────────
# SHARED DRAWING HELPERS
# ─────────────────────────────────────────────

def draw_idle(highlight_idx=None):
    """Draw grid in idle (black) state, optional green highlight."""
    for i, rect in enumerate(stim_rects):
        rect.fillColor = [-1, -1, -1]
        rect.draw()
    for txt in label_texts:
        txt.draw()
    if highlight_idx is not None:
        h = create_highlight_rect(cell_positions[highlight_idx])
        h.draw()
    photosensor_dot.color = [-1, -1, -1]
    photosensor_dot.draw()


def draw_stimulus_frame(i_frame, highlight_idx=None):
    """Draw one flickering frame."""
    for i, rect in enumerate(stim_rects):
        v = stimulus_frames[i_frame, i]
        rect.fillColor = [v, v, v]
        rect.draw()
    for txt in label_texts:
        txt.draw()
    if highlight_idx is not None:
        h = create_highlight_rect(cell_positions[highlight_idx])
        h.draw()
    photosensor_dot.color = [1, 1, 1]
    photosensor_dot.draw()


# ─────────────────────────────────────────────
# EEG COLLECTION HELPER
# ─────────────────────────────────────────────

def collect_trial_eeg(i_trial, skip_count, eeg, aux, timestamp,
                       trial_starts, trial_ends):
    """
    Block until the current trial's EEG is available via photosensor trigger.
    Returns updated (eeg, aux, timestamp, trial_starts, trial_ends, trial_eeg_cropped).
    """
    import mne as _mne
    baseline_dur     = 0.2
    baseline_samples = int(baseline_dur * sampling_rate)
    trial_dur_samp   = int(stim_duration * sampling_rate) + baseline_samples

    while len(trial_ends) <= i_trial + skip_count:
        while not eeg_queue.empty():
            eeg_in, aux_in, ts_in = eeg_queue.get()
            eeg       = np.concatenate((eeg, eeg_in), axis=1)
            aux       = np.concatenate((aux, aux_in), axis=1)
            timestamp = np.concatenate((timestamp, ts_in))
        photo_trigger = (aux[1] > 20).astype(int)
        trial_starts  = np.where(np.diff(photo_trigger) == 1)[0]
        trial_ends    = np.where(np.diff(photo_trigger) == -1)[0]

    t_start      = trial_starts[i_trial + skip_count] - baseline_samples
    filtered_eeg = _mne.filter.filter_data(
        eeg, sfreq=sampling_rate, l_freq=2, h_freq=40, verbose=False)
    trial_eeg    = np.copy(filtered_eeg[:, t_start:t_start + trial_dur_samp])
    trial_aux    = np.copy(aux[:, t_start:t_start + trial_dur_samp])

    baseline_avg = np.mean(trial_eeg[:, :baseline_samples], axis=1, keepdims=True)
    trial_eeg   -= baseline_avg
    cropped      = trial_eeg[:, baseline_samples:]
    return eeg, aux, timestamp, trial_starts, trial_ends, cropped, trial_aux


# ─────────────────────────────────────────────
# CALIBRATION MODE
# ─────────────────────────────────────────────

def run_calibration():
    """
    Show each of the 8 targets in random order (n_per_class repetitions).
    Highlights the target cell so the user knows where to gaze.
    Saves raw EEG + trial data to disk.
    """
    trial_sequence = np.tile(np.arange(N_CLASSES), n_per_class)
    np.random.seed(run)
    np.random.shuffle(trial_sequence)

    eeg       = np.zeros((8, 0))
    aux       = np.zeros((3, 0))
    timestamp = np.zeros(0)
    eeg_trials, aux_trials = [], []
    trial_starts, trial_ends = [], []
    skip_count = 0

    n_trials = len(trial_sequence)

    for i_trial, target_id in enumerate(trial_sequence):
        action = ACTIONS[target_id]
        # ── Inter-trial: show idle grid with target highlighted ──
        info_text = visual.TextStim(
            window,
            text=f'Trial {i_trial+1}/{n_trials}  |  Gaze at: {action["label"].replace(chr(10)," ")}',
            pos=(0, -0.88),
            color='white',
            height=0.07,
            units='norm',
        )
        draw_idle(highlight_idx=target_id)
        info_text.draw()
        window.flip()
        core.wait(0.8)

        # ── Stimulus presentation ──
        for i_frame in range(num_frames):
            keys = keyboard.getKeys()
            if 'escape' in [k.name for k in keys]:
                _emergency_save(eeg, aux, eeg_trials, aux_trials)
                core.quit()

            draw_stimulus_frame(i_frame, highlight_idx=target_id)
            if core.getTime() > window.getFutureFlipTime() and i_frame != 0:
                print(f'[WARNING] Missed frame at trial {i_trial}, frame {i_frame}')
            window.flip()

        # ── Post-stim blank ──
        draw_idle(highlight_idx=target_id)
        window.flip()

        # ── Collect EEG ──
        if cyton_in:
            eeg, aux, timestamp, trial_starts, trial_ends, cropped, trial_aux = \
                collect_trial_eeg(i_trial, skip_count, eeg, aux, timestamp,
                                  trial_starts, trial_ends)
            eeg_trials.append(cropped)
            aux_trials.append(trial_aux)
            print(f'Trial {i_trial+1}/{n_trials} | target={target_id} '
                  f'({action["label"].replace(chr(10)," ")}) | EEG: {cropped.shape}')

    # ── Save ──
    _emergency_save(eeg, aux, eeg_trials, aux_trials)
    print('Calibration complete. Data saved to', save_dir)


def _emergency_save(eeg, aux, eeg_trials, aux_trials):
    if cyton_in:
        np.save(save_file_eeg, eeg)
        np.save(save_file_aux, aux)
        np.save(save_file_eeg_trials, np.array(eeg_trials, dtype=object))
        np.save(save_file_aux_trials, np.array(aux_trials, dtype=object))
        stop_event.set()
        board.stop_stream()
        board.release_session()


# ─────────────────────────────────────────────
# LIVE PLAY MODE
# ─────────────────────────────────────────────

def run_live_play():
    """
    Continuously present all 8 flickering stimuli, predict which the user gazes at,
    and execute the corresponding Hollow Knight key action.
    """
    eeg       = np.zeros((8, 0))
    aux       = np.zeros((3, 0))
    timestamp = np.zeros(0)
    eeg_trials, aux_trials = [], []
    trial_starts, trial_ends = [], []
    skip_count   = 0
    prediction   = 0
    prev_pred    = -1

    # Status text shown on the overlay
    status_text = visual.TextStim(
        window,
        text='',
        pos=(0, -0.88),
        color=[0.8, 0.8, 0.8],
        height=0.07,
        units='norm',
    )

    for i_trial in range(10000):
        action_name = ACTIONS[prediction]['label'].replace('\n', ' ')
        status_text.text = f'Predicted: {action_name}  (trial {i_trial})'

        # ── Inter-trial idle ──
        draw_idle(highlight_idx=prediction)
        status_text.draw()
        window.flip()
        core.wait(0.5)

        # ── Stimulus ──
        for i_frame in range(num_frames):
            keys = keyboard.getKeys()
            if 'escape' in [k.name for k in keys]:
                release_held_keys()
                if cyton_in:
                    stop_event.set()
                    board.stop_stream()
                    board.release_session()
                core.quit()

            draw_stimulus_frame(i_frame, highlight_idx=prediction)
            if core.getTime() > window.getFutureFlipTime() and i_frame != 0:
                print(f'[WARNING] Missed frame at trial {i_trial}, frame {i_frame}')
            window.flip()

        # ── Post-stim blank ──
        draw_idle(highlight_idx=prediction)
        window.flip()

        # ── Collect EEG & predict ──
        if cyton_in and model is not None:
            eeg, aux, timestamp, trial_starts, trial_ends, cropped, trial_aux = \
                collect_trial_eeg(i_trial, skip_count, eeg, aux, timestamp,
                                  trial_starts, trial_ends)
            eeg_trials.append(cropped)
            aux_trials.append(trial_aux)
            prev_pred  = prediction
            prediction = int(model.predict(cropped)[0])
            print(f'Trial {i_trial} | prediction={prediction} '
                  f'({ACTIONS[prediction]["label"].replace(chr(10)," ")})')
            execute_action(prediction, prev_pred)
        else:
            # No model: cycle through targets for testing layout
            prev_pred  = prediction
            prediction = (prediction + 1) % N_CLASSES

    release_held_keys()
    if cyton_in:
        stop_event.set()
        board.stop_stream()
        board.release_session()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    try:
        if calibration_mode:
            run_calibration()
        else:
            run_live_play()
    except Exception as e:
        release_held_keys()
        raise e
    finally:
        window.close()
        core.quit()
