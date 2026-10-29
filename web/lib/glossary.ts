// Plain-language explanations shown on hover. Add an entry here when you add a model.

export const MODEL_INFO: Record<string, string> = {
  hsemotion_b0:
    'A small, fast emotion model trained on ~400,000 internet photos of faces (AffectNet), each tagged with one of ' +
    '8 emotions, then fine-tuned on group videos. It reacts strongly to mouth shapes, so people who are just talking ' +
    'often get read as emotional.',
  hsemotion_b2:
    'A bigger sibling of hsemotion_b0, trained on the same photo collection with 8 emotions. Bigger is not better ' +
    'here: it is the least sure of itself and most often calls a calm face emotional.',
  vit_fer:
    'A different kind of model (a "vision transformer") trained on ~35,000 small black-and-white face photos (FER2013) ' +
    'with 7 emotions — it has no "contempt". So far it matches the human labels best and is the most willing to say ' +
    '"neutral".',
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
    'found. The number is the average strength of that top emotion — low numbers (e.g. 0.3) mean the model was torn.',
  faceVisible: 'Share of frames where a face was found. Frames without a face get no guess at all.',
  timeline:
    'One strip per model. Each thin slice is one frame, coloured by that model\'s top emotion. Long single-colour ' +
    'stretches mean steady guesses; confetti means it keeps changing its mind. Click anywhere to jump there.',
  fps: 'The datasets do not record the original frame rate, so videos are rebuilt at 25 frames per second.',
};
