// Plain-language explanations shown on hover. Add an entry here when you add a model.

export const MODEL_INFO: Record<string, string> = {
  hsemotion_b0:
    'A small, fast emotion model (EfficientNet-B0) trained on AffectNet — hundreds of thousands of internet face ' +
    'photos, each tagged with one of 8 emotions. This is the version its authors also report on group videos (VGAF). ' +
    'It reacts strongly to mouth shapes, so people who are just talking often get read as emotional.',
  hsemotion_b2:
    'A bigger sibling of hsemotion_b0, trained on the same photo collection with 8 emotions. Bigger is not better ' +
    'here: it is the least sure of itself and most often calls a calm face emotional.',
  vit_fer:
    'A different kind of model (a "vision transformer") trained on ~35,000 small black-and-white face photos (FER2013) ' +
    'with 7 emotions — it has no "contempt". So far it matches the human labels best and is the most willing to say ' +
    '"neutral".',
  cmet_mead:
    'The emotion checker that C-MET (a 2026 emotion-editing model) uses to grade its own videos. It looks at each ' +
    'frame of the face and was trained on MEAD — actors reading sentences in 7 acted emotions. It has NO "neutral": ' +
    'it always picks an emotion, so on calm faces its answer is a forced guess.',
  cmet_crema:
    'C-MET\'s second emotion checker, trained on CREMA-D — actors saying short sentences in different moods. It only ' +
    'knows 5 emotions (happy, angry, disgusted, fear, sad): no neutral, no surprise, no contempt, so it is forced ' +
    'to pick one of those five.',
  seamless_imitator:
    'Emotion scores that came with the Seamless dataset itself, made by Meta\'s own model when the dataset was ' +
    'built. Not run by us. We checked its emotion order against hsemotion_b0 before showing it.',
  speech_emotion2vec:
    'Listens to the VOICE, not the face: emotion2vec+, the speech-emotion model C-MET uses to read emotion from ' +
    'audio. It hears 2-second stretches of sound, so its answer changes more slowly than the face models\'. ' +
    'Silence tends to come out as "neutral".',
};

export const modelInfo = (name: string) =>
  MODEL_INFO[name] ?? `No description for "${name}" yet — add one in web/lib/glossary.ts.`;

export const TERMS = {
  overlay:
    'Choose whose guess is drawn on the video. The box marks where the face is; the label is that model\'s guess for ' +
    'the frame on screen, with a score from 0 to 1 for how strongly it sees that emotion.',
  all:
    'Show every model\'s guess at the same time, stacked above the face, so you can see where they agree or disagree.',
  humanLabel:
    'What a person decided this clip shows overall, after watching it. This is the answer key the models are graded ' +
    'against. It is one label for the whole clip, not for each frame.',
  confidence: {
    high: 'The person labelling was very sure: the emotion is clear and obvious.',
    medium: 'The person labelling was fairly sure, but the emotion is not strong.',
    low: 'The person labelling was unsure: the expression is subtle or ambiguous. Treat this label with caution.',
  } as Record<string, string>,
  notes: 'Why the person chose this label, in their own words.',
  frame:
    'The selected model\'s scores for just the frame on screen. Each bar is how strongly it sees that emotion; ' +
    'all bars add up to 1. Pause the video to read a single moment.',
  wholeClip:
    'Each model\'s overall verdict for the clip: its frame-by-frame scores averaged over every frame where a face was ' +
    'found (voice models instead listen to the whole clip\'s audio at once). The number is the strength of that top ' +
    'emotion — low numbers (e.g. 0.3) mean the model was torn.',
  faceVisible: 'Share of frames where a face was found. Frames without a face get no guess at all.',
  timeline:
    'One strip per model. Each thin slice is one frame, coloured by that model\'s top emotion. Long single-colour ' +
    'stretches mean steady guesses; confetti means it keeps changing its mind. Click anywhere to jump there.',
  fps: 'The datasets do not record the original frame rate, so videos are rebuilt at 25 frames per second.',
  transcript: 'What the person is saying right now (from the dataset\'s transcript). The highlighted word is the one ' +
    'being spoken. Useful for checking: does the face match the words?',
  voiceFrame: 'The voice model\'s scores for the 2 seconds of sound around this moment. Each bar is how strongly it ' +
    'hears that emotion; the bars add up to 1.',
  cannotSay: 'This model was never taught this emotion, so it can never answer it. Compare such models on the ' +
    'emotions they know (see "non-neutral" score on the leaderboard).',
  thisFrameAll: 'Every model\'s answer for the frame on screen, side by side. Each coloured bar is one model\'s full ' +
    'breakdown: the wider a colour, the more strongly it sees that emotion. The chip on the right is its top pick.',
  frameGrid: 'One row per model, one column per frame (thumbnails on top). Each cell shows the model\'s top emotion ' +
    '(3-letter code) and how sure it is; stronger colour = more sure. Hover a cell for all scores, click to jump ' +
    'the video there. "—" means no face (or no sound, for voice models) in that frame.',
  dataWarning: 'A small mismatch in the source data for this clip (for example one frame more than the dataset\'s ' +
    'timing table says). The clip is still shown; numbers may be off by one frame.',
};
