#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.3),
    on November 15, 2023, at 15:09
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.3'
expName = 'experience'  # from the Builder filename that created this script
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Master\\PC\\ColoredDobble\\experience_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=(1024, 768), fullscr=True, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height'
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    win.mouseVisible = False
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions" ---
    text = visual.TextStim(win=win, name='text',
        text="Welcome to Colored Dobble!\n\nThis is a color-shade discrimination experiment.\n\nThere will be a small test at the beginning so you can understand the experiment better.\n\nYou will be given 3 color-shades in the center of your screen.\n\nThen just below, you will have 3 possible answers.\n\nYou should choose the CORRECT color shade that matches one of the 3 color shades from the center of the screen.\n\nThere is always ONE correct answer.\n\nUse LEFT, UP and RIGHT arrow keys to choose one of the 3 possible answers.\n\nDo NOT answer RANDOMLY. If you don't see the difference between colors or are not sure about the answer, press the DOWN arrow key.\n\nPress SPACEBAR to start the test.\n\nPeace.",
        font='Open Sans',
        pos=(0, 0), height=0.03, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "examples" ---
    polygon = visual.ShapeStim(
        win=win, name='polygon',
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(-0.3, 0.0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, 1.0000, 1.0000], fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    polygon_2 = visual.ShapeStim(
        win=win, name='polygon_2',
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(0.3, 0.0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    polygon_3 = visual.ShapeStim(
        win=win, name='polygon_3',
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(0, 0.3), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    answer_1 = visual.ShapeStim(
        win=win, name='answer_1',
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(-0.4, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    answer_2 = visual.ShapeStim(
        win=win, name='answer_2',
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(0, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    answer_3 = visual.ShapeStim(
        win=win, name='answer_3',
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(0.4, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    key_resp = keyboard.Keyboard()
    prog_2 = visual.Progress(
        win, name='prog_2',
        progress=0.0,
        pos=(-0.5, 0.5), size=(1, 0.03), anchor='top-left', units=win.units,
        barColor='white', backColor=None, borderColor='white', colorSpace='rgb',
        lineWidth=4.0, opacity=1.0, ori=0.0,
        depth=-7
    )
    arrow_left = visual.ShapeStim(
        win=win, name='arrow_left', vertices='arrow',
        size=(0.05, 0.05),
        ori=-90.0, pos=(-0.4, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-8.0, interpolate=True)
    arrow_up = visual.ShapeStim(
        win=win, name='arrow_up', vertices='arrow',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-9.0, interpolate=True)
    arrow_right = visual.ShapeStim(
        win=win, name='arrow_right', vertices='arrow',
        size=(0.05, 0.05),
        ori=90.0, pos=(0.4, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-10.0, interpolate=True)
    text_3 = visual.TextStim(win=win, name='text_3',
        text='yo need to answer in these 5 seconds bro :)',
        font='Open Sans',
        pos=(0, 0.4), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    
    # --- Initialize components for Routine "begining_exper" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Now, you have 21 REAL questions to answer.\n\nRemember, there are only 5 seconds per question.\n\nIf you are not sure for the answer, press DOWN.\n\nPress SPACEBAR to start.',
        font='Open Sans',
        pos=(0, 0), height=0.04, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "trials_5" ---
    circle1 = visual.ShapeStim(
        win=win, name='circle1',
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(-0.3, 0.0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[1.0000, 1.0000, 1.0000], fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    circle2 = visual.ShapeStim(
        win=win, name='circle2',
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(0.3, 0.0), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-1.0, interpolate=True)
    circle3 = visual.ShapeStim(
        win=win, name='circle3',
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(0, 0.3), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    answerLeft = visual.ShapeStim(
        win=win, name='answerLeft',
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(-0.4, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    answerUp = visual.ShapeStim(
        win=win, name='answerUp',
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(0, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-4.0, interpolate=True)
    answerRight = visual.ShapeStim(
        win=win, name='answerRight',
        size=(0.2, 0.2), vertices='circle',
        ori=0.0, pos=(0.4, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
        opacity=None, depth=-5.0, interpolate=True)
    keysUsed = keyboard.Keyboard()
    arrowLeft = visual.ShapeStim(
        win=win, name='arrowLeft', vertices='arrow',
        size=(0.05, 0.05),
        ori=-90.0, pos=(-0.4, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-7.0, interpolate=True)
    arrowUp = visual.ShapeStim(
        win=win, name='arrowUp', vertices='arrow',
        size=(0.05, 0.05),
        ori=0.0, pos=(0, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-8.0, interpolate=True)
    arrowRight = visual.ShapeStim(
        win=win, name='arrowRight', vertices='arrow',
        size=(0.05, 0.05),
        ori=90.0, pos=(0.4, -0.4), anchor='center',
        lineWidth=1.0,     colorSpace='rgb',  lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-9.0, interpolate=True)
    
    # --- Initialize components for Routine "thanks_message" ---
    text_4 = visual.TextStim(win=win, name='text_4',
        text='thank you for your time!!!\n\npress space to exit!',
        font='Open Sans',
        pos=(0, 0), height=0.07, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_4 = keyboard.Keyboard()
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "instructions" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructions.started', globalClock.getTime())
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # keep track of which components have finished
    instructionsComponents = [text, key_resp_2]
    for thisComponent in instructionsComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            key_resp_2.clock.reset()  # now t=0
        if key_resp_2.status == STARTED:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionsComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructions.stopped', globalClock.getTime())
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_3 = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('HEX.xlsx'),
        seed=None, name='trials_3')
    thisExp.addLoop(trials_3)  # add the loop to the experiment
    thisTrial_3 = trials_3.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
    if thisTrial_3 != None:
        for paramName in thisTrial_3:
            globals()[paramName] = thisTrial_3[paramName]
    
    for thisTrial_3 in trials_3:
        currentLoop = trials_3
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_3.rgb)
        if thisTrial_3 != None:
            for paramName in thisTrial_3:
                globals()[paramName] = thisTrial_3[paramName]
        
        # --- Prepare to start Routine "examples" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('examples.started', globalClock.getTime())
        polygon.setFillColor(HEX1)
        polygon_2.setFillColor(HEX2)
        polygon_3.setFillColor(HEX3)
        answer_1.setFillColor(HEXL)
        answer_2.setFillColor(HEXU)
        answer_3.setFillColor(HEXR)
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        prog_2.setProgress(prog)
        # keep track of which components have finished
        examplesComponents = [polygon, polygon_2, polygon_3, answer_1, answer_2, answer_3, key_resp, prog_2, arrow_left, arrow_up, arrow_right, text_3]
        for thisComponent in examplesComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "examples" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *polygon* updates
            
            # if polygon is starting this frame...
            if polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon.frameNStart = frameN  # exact frame index
                polygon.tStart = t  # local t and not account for scr refresh
                polygon.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon, 'tStartRefresh')  # time at next scr refresh
                # update status
                polygon.status = STARTED
                polygon.setAutoDraw(True)
            
            # if polygon is active this frame...
            if polygon.status == STARTED:
                # update params
                pass
            
            # if polygon is stopping this frame...
            if polygon.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon.tStop = t  # not accounting for scr refresh
                    polygon.frameNStop = frameN  # exact frame index
                    # update status
                    polygon.status = FINISHED
                    polygon.setAutoDraw(False)
            
            # *polygon_2* updates
            
            # if polygon_2 is starting this frame...
            if polygon_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_2.frameNStart = frameN  # exact frame index
                polygon_2.tStart = t  # local t and not account for scr refresh
                polygon_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                polygon_2.status = STARTED
                polygon_2.setAutoDraw(True)
            
            # if polygon_2 is active this frame...
            if polygon_2.status == STARTED:
                # update params
                pass
            
            # if polygon_2 is stopping this frame...
            if polygon_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon_2.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_2.tStop = t  # not accounting for scr refresh
                    polygon_2.frameNStop = frameN  # exact frame index
                    # update status
                    polygon_2.status = FINISHED
                    polygon_2.setAutoDraw(False)
            
            # *polygon_3* updates
            
            # if polygon_3 is starting this frame...
            if polygon_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                polygon_3.frameNStart = frameN  # exact frame index
                polygon_3.tStart = t  # local t and not account for scr refresh
                polygon_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(polygon_3, 'tStartRefresh')  # time at next scr refresh
                # update status
                polygon_3.status = STARTED
                polygon_3.setAutoDraw(True)
            
            # if polygon_3 is active this frame...
            if polygon_3.status == STARTED:
                # update params
                pass
            
            # if polygon_3 is stopping this frame...
            if polygon_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > polygon_3.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    polygon_3.tStop = t  # not accounting for scr refresh
                    polygon_3.frameNStop = frameN  # exact frame index
                    # update status
                    polygon_3.status = FINISHED
                    polygon_3.setAutoDraw(False)
            
            # *answer_1* updates
            
            # if answer_1 is starting this frame...
            if answer_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                answer_1.frameNStart = frameN  # exact frame index
                answer_1.tStart = t  # local t and not account for scr refresh
                answer_1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(answer_1, 'tStartRefresh')  # time at next scr refresh
                # update status
                answer_1.status = STARTED
                answer_1.setAutoDraw(True)
            
            # if answer_1 is active this frame...
            if answer_1.status == STARTED:
                # update params
                pass
            
            # *answer_2* updates
            
            # if answer_2 is starting this frame...
            if answer_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                answer_2.frameNStart = frameN  # exact frame index
                answer_2.tStart = t  # local t and not account for scr refresh
                answer_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(answer_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                answer_2.status = STARTED
                answer_2.setAutoDraw(True)
            
            # if answer_2 is active this frame...
            if answer_2.status == STARTED:
                # update params
                pass
            
            # *answer_3* updates
            
            # if answer_3 is starting this frame...
            if answer_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                answer_3.frameNStart = frameN  # exact frame index
                answer_3.tStart = t  # local t and not account for scr refresh
                answer_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(answer_3, 'tStartRefresh')  # time at next scr refresh
                # update status
                answer_3.status = STARTED
                answer_3.setAutoDraw(True)
            
            # if answer_3 is active this frame...
            if answer_3.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['left','right','up','down'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[0].name  # just the first key pressed
                    key_resp.rt = _key_resp_allKeys[0].rt
                    key_resp.duration = _key_resp_allKeys[0].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *prog_2* updates
            
            # if prog_2 is starting this frame...
            if prog_2.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                prog_2.frameNStart = frameN  # exact frame index
                prog_2.tStart = t  # local t and not account for scr refresh
                prog_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(prog_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                prog_2.status = STARTED
                prog_2.setAutoDraw(True)
            
            # if prog_2 is active this frame...
            if prog_2.status == STARTED:
                # update params
                pass
            
            # *arrow_left* updates
            
            # if arrow_left is starting this frame...
            if arrow_left.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                arrow_left.frameNStart = frameN  # exact frame index
                arrow_left.tStart = t  # local t and not account for scr refresh
                arrow_left.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrow_left, 'tStartRefresh')  # time at next scr refresh
                # update status
                arrow_left.status = STARTED
                arrow_left.setAutoDraw(True)
            
            # if arrow_left is active this frame...
            if arrow_left.status == STARTED:
                # update params
                pass
            
            # *arrow_up* updates
            
            # if arrow_up is starting this frame...
            if arrow_up.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                arrow_up.frameNStart = frameN  # exact frame index
                arrow_up.tStart = t  # local t and not account for scr refresh
                arrow_up.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrow_up, 'tStartRefresh')  # time at next scr refresh
                # update status
                arrow_up.status = STARTED
                arrow_up.setAutoDraw(True)
            
            # if arrow_up is active this frame...
            if arrow_up.status == STARTED:
                # update params
                pass
            
            # *arrow_right* updates
            
            # if arrow_right is starting this frame...
            if arrow_right.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                arrow_right.frameNStart = frameN  # exact frame index
                arrow_right.tStart = t  # local t and not account for scr refresh
                arrow_right.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrow_right, 'tStartRefresh')  # time at next scr refresh
                # update status
                arrow_right.status = STARTED
                arrow_right.setAutoDraw(True)
            
            # if arrow_right is active this frame...
            if arrow_right.status == STARTED:
                # update params
                pass
            
            # *text_3* updates
            
            # if text_3 is starting this frame...
            if text_3.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # update status
                text_3.status = STARTED
                text_3.setAutoDraw(True)
            
            # if text_3 is active this frame...
            if text_3.status == STARTED:
                # update params
                pass
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in examplesComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "examples" ---
        for thisComponent in examplesComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('examples.stopped', globalClock.getTime())
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trials_3.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trials_3.addData('key_resp.rt', key_resp.rt)
            trials_3.addData('key_resp.duration', key_resp.duration)
        # the Routine "examples" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'trials_3'
    
    
    # --- Prepare to start Routine "begining_exper" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('begining_exper.started', globalClock.getTime())
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # keep track of which components have finished
    begining_experComponents = [text_2, key_resp_3]
    for thisComponent in begining_experComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "begining_exper" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            key_resp_3.clock.reset()  # now t=0
        if key_resp_3.status == STARTED:
            theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in begining_experComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "begining_exper" ---
    for thisComponent in begining_experComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('begining_exper.stopped', globalClock.getTime())
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "begining_exper" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    L_varied = data.TrialHandler(nReps=1.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('HEX_ALL.xlsx'),
        seed=None, name='L_varied')
    thisExp.addLoop(L_varied)  # add the loop to the experiment
    thisL_varied = L_varied.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisL_varied.rgb)
    if thisL_varied != None:
        for paramName in thisL_varied:
            globals()[paramName] = thisL_varied[paramName]
    
    for thisL_varied in L_varied:
        currentLoop = L_varied
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisL_varied.rgb)
        if thisL_varied != None:
            for paramName in thisL_varied:
                globals()[paramName] = thisL_varied[paramName]
        
        # --- Prepare to start Routine "trials_5" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('trials_5.started', globalClock.getTime())
        circle1.setFillColor(HEX1)
        circle2.setFillColor(HEX2)
        circle3.setFillColor(HEX3)
        answerLeft.setFillColor(HEXL)
        answerUp.setFillColor(HEXU)
        answerRight.setFillColor(HEXR)
        keysUsed.keys = []
        keysUsed.rt = []
        _keysUsed_allKeys = []
        # keep track of which components have finished
        trials_5Components = [circle1, circle2, circle3, answerLeft, answerUp, answerRight, keysUsed, arrowLeft, arrowUp, arrowRight]
        for thisComponent in trials_5Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trials_5" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *circle1* updates
            
            # if circle1 is starting this frame...
            if circle1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                circle1.frameNStart = frameN  # exact frame index
                circle1.tStart = t  # local t and not account for scr refresh
                circle1.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circle1, 'tStartRefresh')  # time at next scr refresh
                # update status
                circle1.status = STARTED
                circle1.setAutoDraw(True)
            
            # if circle1 is active this frame...
            if circle1.status == STARTED:
                # update params
                pass
            
            # if circle1 is stopping this frame...
            if circle1.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > circle1.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    circle1.tStop = t  # not accounting for scr refresh
                    circle1.frameNStop = frameN  # exact frame index
                    # update status
                    circle1.status = FINISHED
                    circle1.setAutoDraw(False)
            
            # *circle2* updates
            
            # if circle2 is starting this frame...
            if circle2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                circle2.frameNStart = frameN  # exact frame index
                circle2.tStart = t  # local t and not account for scr refresh
                circle2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circle2, 'tStartRefresh')  # time at next scr refresh
                # update status
                circle2.status = STARTED
                circle2.setAutoDraw(True)
            
            # if circle2 is active this frame...
            if circle2.status == STARTED:
                # update params
                pass
            
            # if circle2 is stopping this frame...
            if circle2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > circle2.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    circle2.tStop = t  # not accounting for scr refresh
                    circle2.frameNStop = frameN  # exact frame index
                    # update status
                    circle2.status = FINISHED
                    circle2.setAutoDraw(False)
            
            # *circle3* updates
            
            # if circle3 is starting this frame...
            if circle3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                circle3.frameNStart = frameN  # exact frame index
                circle3.tStart = t  # local t and not account for scr refresh
                circle3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(circle3, 'tStartRefresh')  # time at next scr refresh
                # update status
                circle3.status = STARTED
                circle3.setAutoDraw(True)
            
            # if circle3 is active this frame...
            if circle3.status == STARTED:
                # update params
                pass
            
            # if circle3 is stopping this frame...
            if circle3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > circle3.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    circle3.tStop = t  # not accounting for scr refresh
                    circle3.frameNStop = frameN  # exact frame index
                    # update status
                    circle3.status = FINISHED
                    circle3.setAutoDraw(False)
            
            # *answerLeft* updates
            
            # if answerLeft is starting this frame...
            if answerLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                answerLeft.frameNStart = frameN  # exact frame index
                answerLeft.tStart = t  # local t and not account for scr refresh
                answerLeft.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(answerLeft, 'tStartRefresh')  # time at next scr refresh
                # update status
                answerLeft.status = STARTED
                answerLeft.setAutoDraw(True)
            
            # if answerLeft is active this frame...
            if answerLeft.status == STARTED:
                # update params
                pass
            
            # if answerLeft is stopping this frame...
            if answerLeft.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > answerLeft.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    answerLeft.tStop = t  # not accounting for scr refresh
                    answerLeft.frameNStop = frameN  # exact frame index
                    # update status
                    answerLeft.status = FINISHED
                    answerLeft.setAutoDraw(False)
            
            # *answerUp* updates
            
            # if answerUp is starting this frame...
            if answerUp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                answerUp.frameNStart = frameN  # exact frame index
                answerUp.tStart = t  # local t and not account for scr refresh
                answerUp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(answerUp, 'tStartRefresh')  # time at next scr refresh
                # update status
                answerUp.status = STARTED
                answerUp.setAutoDraw(True)
            
            # if answerUp is active this frame...
            if answerUp.status == STARTED:
                # update params
                pass
            
            # if answerUp is stopping this frame...
            if answerUp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > answerUp.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    answerUp.tStop = t  # not accounting for scr refresh
                    answerUp.frameNStop = frameN  # exact frame index
                    # update status
                    answerUp.status = FINISHED
                    answerUp.setAutoDraw(False)
            
            # *answerRight* updates
            
            # if answerRight is starting this frame...
            if answerRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                answerRight.frameNStart = frameN  # exact frame index
                answerRight.tStart = t  # local t and not account for scr refresh
                answerRight.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(answerRight, 'tStartRefresh')  # time at next scr refresh
                # update status
                answerRight.status = STARTED
                answerRight.setAutoDraw(True)
            
            # if answerRight is active this frame...
            if answerRight.status == STARTED:
                # update params
                pass
            
            # if answerRight is stopping this frame...
            if answerRight.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > answerRight.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    answerRight.tStop = t  # not accounting for scr refresh
                    answerRight.frameNStop = frameN  # exact frame index
                    # update status
                    answerRight.status = FINISHED
                    answerRight.setAutoDraw(False)
            
            # *keysUsed* updates
            waitOnFlip = False
            
            # if keysUsed is starting this frame...
            if keysUsed.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                keysUsed.frameNStart = frameN  # exact frame index
                keysUsed.tStart = t  # local t and not account for scr refresh
                keysUsed.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(keysUsed, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'keysUsed.started')
                # update status
                keysUsed.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(keysUsed.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(keysUsed.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if keysUsed is stopping this frame...
            if keysUsed.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > keysUsed.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    keysUsed.tStop = t  # not accounting for scr refresh
                    keysUsed.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'keysUsed.stopped')
                    # update status
                    keysUsed.status = FINISHED
                    keysUsed.status = FINISHED
            if keysUsed.status == STARTED and not waitOnFlip:
                theseKeys = keysUsed.getKeys(keyList=['left','right','up','down'], ignoreKeys=["escape"], waitRelease=False)
                _keysUsed_allKeys.extend(theseKeys)
                if len(_keysUsed_allKeys):
                    keysUsed.keys = [key.name for key in _keysUsed_allKeys]  # storing all keys
                    keysUsed.rt = [key.rt for key in _keysUsed_allKeys]
                    keysUsed.duration = [key.duration for key in _keysUsed_allKeys]
                    # a response ends the routine
                    continueRoutine = False
            
            # *arrowLeft* updates
            
            # if arrowLeft is starting this frame...
            if arrowLeft.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                arrowLeft.frameNStart = frameN  # exact frame index
                arrowLeft.tStart = t  # local t and not account for scr refresh
                arrowLeft.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrowLeft, 'tStartRefresh')  # time at next scr refresh
                # update status
                arrowLeft.status = STARTED
                arrowLeft.setAutoDraw(True)
            
            # if arrowLeft is active this frame...
            if arrowLeft.status == STARTED:
                # update params
                pass
            
            # if arrowLeft is stopping this frame...
            if arrowLeft.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > arrowLeft.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    arrowLeft.tStop = t  # not accounting for scr refresh
                    arrowLeft.frameNStop = frameN  # exact frame index
                    # update status
                    arrowLeft.status = FINISHED
                    arrowLeft.setAutoDraw(False)
            
            # *arrowUp* updates
            
            # if arrowUp is starting this frame...
            if arrowUp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                arrowUp.frameNStart = frameN  # exact frame index
                arrowUp.tStart = t  # local t and not account for scr refresh
                arrowUp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrowUp, 'tStartRefresh')  # time at next scr refresh
                # update status
                arrowUp.status = STARTED
                arrowUp.setAutoDraw(True)
            
            # if arrowUp is active this frame...
            if arrowUp.status == STARTED:
                # update params
                pass
            
            # if arrowUp is stopping this frame...
            if arrowUp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > arrowUp.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    arrowUp.tStop = t  # not accounting for scr refresh
                    arrowUp.frameNStop = frameN  # exact frame index
                    # update status
                    arrowUp.status = FINISHED
                    arrowUp.setAutoDraw(False)
            
            # *arrowRight* updates
            
            # if arrowRight is starting this frame...
            if arrowRight.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                arrowRight.frameNStart = frameN  # exact frame index
                arrowRight.tStart = t  # local t and not account for scr refresh
                arrowRight.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(arrowRight, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'arrowRight.started')
                # update status
                arrowRight.status = STARTED
                arrowRight.setAutoDraw(True)
            
            # if arrowRight is active this frame...
            if arrowRight.status == STARTED:
                # update params
                pass
            
            # if arrowRight is stopping this frame...
            if arrowRight.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > arrowRight.tStartRefresh + 5-frameTolerance:
                    # keep track of stop time/frame for later
                    arrowRight.tStop = t  # not accounting for scr refresh
                    arrowRight.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'arrowRight.stopped')
                    # update status
                    arrowRight.status = FINISHED
                    arrowRight.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trials_5Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trials_5" ---
        for thisComponent in trials_5Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('trials_5.stopped', globalClock.getTime())
        # check responses
        if keysUsed.keys in ['', [], None]:  # No response was made
            keysUsed.keys = None
        L_varied.addData('keysUsed.keys',keysUsed.keys)
        if keysUsed.keys != None:  # we had a response
            L_varied.addData('keysUsed.rt', keysUsed.rt)
            L_varied.addData('keysUsed.duration', keysUsed.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 1.0 repeats of 'L_varied'
    
    
    # --- Prepare to start Routine "thanks_message" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('thanks_message.started', globalClock.getTime())
    key_resp_4.keys = []
    key_resp_4.rt = []
    _key_resp_4_allKeys = []
    # keep track of which components have finished
    thanks_messageComponents = [text_4, key_resp_4]
    for thisComponent in thanks_messageComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thanks_message" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # *key_resp_4* updates
        waitOnFlip = False
        
        # if key_resp_4 is starting this frame...
        if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_4.frameNStart = frameN  # exact frame index
            key_resp_4.tStart = t  # local t and not account for scr refresh
            key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_4.started')
            # update status
            key_resp_4.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_4.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_4.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_4_allKeys.extend(theseKeys)
            if len(_key_resp_4_allKeys):
                key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thanks_messageComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thanks_message" ---
    for thisComponent in thanks_messageComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('thanks_message.stopped', globalClock.getTime())
    # check responses
    if key_resp_4.keys in ['', [], None]:  # No response was made
        key_resp_4.keys = None
    thisExp.addData('key_resp_4.keys',key_resp_4.keys)
    if key_resp_4.keys != None:  # we had a response
        thisExp.addData('key_resp_4.rt', key_resp_4.rt)
        thisExp.addData('key_resp_4.duration', key_resp_4.duration)
    thisExp.nextEntry()
    # the Routine "thanks_message" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
