// ============================================================
// BubbleSem v2 — Two-Section Experiment
//
// Section 1 (Baseline): some_masked + most_masked trials shuffled
//   - All 20 target words shown with 20% or 40% of nonce words revealed
//   - No reveal interactivity; participant makes guess when ready
//
// Section 2 (Open-Ended): longer passages loaded from open_ended_passages.csv
//   - Participant reads each passage and answers an open-ended question
//     about what the passage is about
//   - Response and timing recorded; no target-word guessing
//
// Note: The prior Section 2 (predetermined trajectory reveal with
//   points system) is preserved in determined_trajectory.js.
//
// CSV files required:
//   trial_lists/sublist_X.csv  — 20 Phase 1 baseline trials (varies by sublist)
//     columns: phase, target_word, passage_variant, jabber_passage,
//              target_word_position, masking_level, unmasked_word_indices,
//              entropy, target_probability, ...
//   open_ended_passages.csv  — Phase 2 longer passages (same for all participants)
//     columns: passage_id, longer_passage
//
// URL parameters:
//   sublist=1..8  (default: 1)
//   subjCode=<string> (default: random ID)
// ============================================================

// ===== GLOBAL STATE =====

let baselineTrialData = [];
let phase2TrialData   = [];
let trialSequenceData = {};   // accumulates data across a trial's screens
let consolidatedTrials = [];  // all saved trial rows
let startTime = null;

// Words that are always shown as real (never masked)
const ARTICLES = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
                  'to', 'for', 'of', 'with', 'by'];

// ===== URL PARAMETERS + SUBJECT CODE =====

function getURLParameter(name) {
    return new URLSearchParams(window.location.search).get(name);
}

const subjCode = getURLParameter('subjCode') || jsPsych.randomization.randomID(10);

let sublistNumber = null;  // set in assignSublist() at timeline start

const seedParam  = getURLParameter('seed');
const randomSeed = seedParam ? parseInt(seedParam) : Math.floor(Math.random() * 1000000);
const filename   = `${subjCode}.csv`;

// ===== JSPSYCH INIT =====

const jsPsych = initJsPsych({});

// ===== SEEDED RNG =====

class SeededRandom {
    constructor(seed) { this.seed = seed; }

    next() {
        this.seed = (this.seed * 9301 + 49297) % 233280;
        return this.seed / 233280;
    }

    shuffle(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(this.next() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }
}

// ===== UTILITY FUNCTIONS =====

// Split a sentence string into word and punctuation tokens.
// e.g. "Hello, world." => ["Hello", ",", "world", "."]
function tokenizeSentence(sentence) {
    const tokens = [];
    sentence.split(' ').forEach(word => {
        const match = word.match(/^([^.,!?;:'"]*)([.,!?;:'"]*)$/);
        if (match) {
            const [, wordPart, punctPart] = match;
            if (wordPart) tokens.push(wordPart);
            if (punctPart) punctPart.split('').forEach(p => tokens.push(p));
        } else {
            tokens.push(word);
        }
    });
    return tokens;
}

// Return true if a token is punctuation.
function isPunct(token) {
    return /^[.,!?;:'"]$/.test(token);
}

// Convert a 0-indexed word position (skipping punctuation tokens) to a token index.
function wordPosToTokenIndex(tokens, wordPos) {
    let wordCount = 0;
    for (let i = 0; i < tokens.length; i++) {
        if (!isPunct(tokens[i])) {
            if (wordCount === wordPos) return i;
            wordCount++;
        }
    }
    console.error(`wordPosToTokenIndex: position ${wordPos} not found (${wordCount} words in tokens)`);
    return -1;
}

// Build a Map from token index → word position (0-indexed, ignoring punctuation).
// Used to convert internal token indices back to word positions when saving data.
function buildTokenToWordPosMap(tokens) {
    const map = new Map();
    let wordPos = 0;
    for (let i = 0; i < tokens.length; i++) {
        if (!isPunct(tokens[i])) {
            map.set(i, wordPos);
            wordPos++;
        }
    }
    return map;
}

// Parse a JSON array column from CSV (PapaParse may leave it as a string).
function parseJSONColumn(value) {
    if (Array.isArray(value)) return value;
    if (typeof value === 'string') {
        try { return JSON.parse(value); }
        catch (e) {
            console.error('parseJSONColumn: failed to parse', value);
            return [];
        }
    }
    return [];
}

// Return true if this token should always be shown as its real English word
// (articles, function words, or words identical in jabberwocky and real versions).
function isAutoRevealed(jabberToken, realToken) {
    const cleanJ = jabberToken.toLowerCase().replace(/[.,!?;:'"]/g, '');
    const cleanR = realToken.toLowerCase().replace(/[.,!?;:'"]/g, '');
    return ARTICLES.includes(cleanJ) || ARTICLES.includes(cleanR) || cleanJ === cleanR;
}

// Return token indices of all maskable words in a passage:
// words that are nonce in jabber but real in the original — i.e. the two tokens differ.
// Excludes the target token index and punctuation.
function getMaskableTokenIndices(jabberTokens, realTokens, targetTokenIdx) {
    const maskable = [];
    for (let i = 0; i < jabberTokens.length; i++) {
        if (i === targetTokenIdx) continue;
        if (isPunct(jabberTokens[i])) continue;
        if (!isAutoRevealed(jabberTokens[i], realTokens[i])) {
            maskable.push(i);
        }
    }
    return maskable;
}

// Convert an array of objects to a CSV string.
// Headers are the union of all keys across every row so that Phase 1 and
// Phase 2 columns all appear even though each phase has unique fields.
function arrayToCSV(data) {
    if (!data.length) return '';
    const headers = [...new Set(data.flatMap(row => Object.keys(row)))];
    const escape = val => {
        if (val === null || val === undefined) return '';
        const str = String(val);
        return (str.includes(',') || str.includes('"') || str.includes('\n'))
            ? `"${str.replace(/"/g, '""')}"` : str;
    };
    return [
        headers.join(','),
        ...data.map(row => headers.map(h => escape(row[h])).join(','))
    ].join('\n');
}

// ===== CONDITION ASSIGNMENT =====

const EXPERIMENT_ID = 'PYSjeESL3lfq';
const N_SUBLISTS    = 8;

// Returns a sublist number (1–8).
// If ?sublist= is in the URL, uses that.
// Otherwise calls the DataPipe condition assignment API for counterbalanced assignment.
async function assignSublist() {
    const param  = getURLParameter('sublist');
    const parsed = parseInt(param);
    if (param && parsed >= 1 && parsed <= N_SUBLISTS) {
        console.log(`Sublist from URL: ${parsed}`);
        return parsed;
    }

    try {
        const resp = await fetch('https://pipe.jspsych.org/api/condition/', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ experiment_id: EXPERIMENT_ID })
        });
        console.log('DataPipe response status:', resp.status);
        if (!resp.ok) {
            console.warn(`DataPipe returned HTTP ${resp.status} — defaulting to sublist 1.`);
            return 1;
        }
        const data = await resp.json();
        console.log('DataPipe response body:', data);
        const conditionNum = parseInt(data.condition);
        if (isNaN(conditionNum)) {
            console.warn('DataPipe returned unexpected condition value:', data, '— defaulting to sublist 1.');
            return 1;
        }
        // DataPipe returns 0-indexed condition; map to 1–N_SUBLISTS
        const assigned = (conditionNum % N_SUBLISTS) + 1;
        console.log(`Sublist from DataPipe condition assignment: ${assigned}`);
        return assigned;
    } catch (err) {
        console.warn('DataPipe condition assignment failed, defaulting to sublist 1.', err);
        return 1;
    }
}

// ===== DATA LOADING =====

function loadCSV(csvFilename) {
    return new Promise((resolve, reject) => {
        Papa.parse(csvFilename, {
            download: true,
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
            complete: results => {
                if (!results.data.length) {
                    reject(new Error(`CSV file is empty: ${csvFilename}`));
                } else {
                    console.log(`Loaded ${results.data.length} rows from ${csvFilename}`);
                    console.log('Sample row:', results.data[0]);
                    resolve(results.data);
                }
            },
            error: err => reject(err)
        });
    });
}

async function loadAllTrialData() {
    const allTrials = await loadCSV(`trial_lists/sublist_${sublistNumber}.csv`);

    const baseline  = allTrials.filter(t => t.condition === 'some_masked' || t.condition === 'most_masked');
    const openEnded = allTrials.filter(t => t.condition === 'open_ended');

    // Shuffle each phase independently with seeded RNG
    const rng = new SeededRandom(randomSeed);
    baselineTrialData = rng.shuffle(baseline);
    phase2TrialData   = rng.shuffle(openEnded);

    console.log(`Baseline trials: ${baselineTrialData.length}`);
    console.log(`Phase 2 trials:  ${phase2TrialData.length}`);
}

// ===== BASELINE TRIAL =====
// Shows the passage with predetermined masked/unmasked words.
// No reveal interactivity — participant clicks "Make Guess" when ready.

function createBaselineTrial(trial, sectionTrialIndex, totalBaseline) {
    const realSentence   = trial.real_passage    || '';
    const jabberSentence = trial.jabber_passage  || '';
    const realTokens     = tokenizeSentence(realSentence);
    const jabberTokens   = tokenizeSentence(jabberSentence);
    // Use jabberTokens for targetTokenIdx — the render loop iterates jabberTokens,
    // so the index must come from the same tokenization.
    const targetTokenIdx = wordPosToTokenIndex(jabberTokens, trial.target_word_position);
    const maskingLevel   = trial.condition || 'some_masked';

    // Words that are always unmasked in every condition (specified in the CSV).
    const alwaysUnmaskedWordPositions = parseJSONColumn(trial.unmasked_word_indices);
    const alwaysUnmaskedTokenIdxSet = new Set(
        alwaysUnmaskedWordPositions
            .map(pos => wordPosToTokenIndex(jabberTokens, pos))
            .filter(i => i >= 0)
    );

    // Revealable pool: maskable words not already in the always-unmasked set.
    // some_masked → randomly unmask 20% of pool; most_masked → 40%.
    const revealFraction = (maskingLevel === 'most_masked') ? 0.40 : 0.20;
    const allMaskableTokenIndices = getMaskableTokenIndices(jabberTokens, realTokens, targetTokenIdx);
    const revealablePool = allMaskableTokenIndices.filter(i => !alwaysUnmaskedTokenIdxSet.has(i));
    const nToReveal = Math.round(revealablePool.length * revealFraction);

    // Derive a per-trial seed from the global seed + trial index so each trial is
    // independently reproducible but all trials are tied to the participant's session.
    const trialRng = new SeededRandom(randomSeed + sectionTrialIndex * 1000);
    const shuffledPool = trialRng.shuffle(revealablePool);
    const randomlyUnmaskedTokenIdxSet = new Set(shuffledPool.slice(0, nToReveal));

    // Combined set: always-unmasked + randomly sampled
    const unmaskedTokenIdxSet = new Set([...alwaysUnmaskedTokenIdxSet, ...randomlyUnmaskedTokenIdxSet]);

    // Record both sets of word positions (0-indexed, punct-excluded) for data output.
    const tokenToWordPos = buildTokenToWordPosMap(jabberTokens);
    const randomlyUnmaskedWordPositions = shuffledPool.slice(0, nToReveal)
        .map(tokIdx => tokenToWordPos.get(tokIdx))
        .filter(pos => pos !== undefined);

    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: function () {
            startTime = Date.now();

            trialSequenceData = {
                subjCode:               subjCode,
                sublist:                sublistNumber,
                random_seed:            randomSeed,
                trial_type:             'baseline',
                trial_number:           trial.trial_number,
                condition:              maskingLevel,
                section_trial_index:    sectionTrialIndex + 1,
                target_word:            trial.target_word,
                target_word_position:   trial.target_word_position,
                entropy:                trial.entropy,
                target_probability:     trial.target_probability,
                real_passage:           realSentence,
                jabber_passage:         jabberSentence,
                always_unmasked_word_indices:   JSON.stringify(alwaysUnmaskedWordPositions),
                randomly_unmasked_word_indices: JSON.stringify(randomlyUnmaskedWordPositions),
            };

            let html = `
                <div class="trial-counter">
                    Section 1 &mdash; Trial ${sectionTrialIndex + 1} of ${totalBaseline}
                </div>
                <div class="sentence-container baseline-passage" id="sentence-container">
            `;

            for (let i = 0; i < jabberTokens.length; i++) {
                const token = jabberTokens[i];

                if (isPunct(token)) {
                    html += token;
                    if (/[.,!?;:]/.test(token) && i < jabberTokens.length - 1) html += ' ';
                    continue;
                }

                if (i === targetTokenIdx) {
                    // Target word: always show as jabberwocky, bold
                    html += `<span class="word target">${token}</span> `;
                } else if (
                    unmaskedTokenIdxSet.has(i) ||
                    isAutoRevealed(jabberTokens[i], realTokens[i])
                ) {
                    // Unmasked: show real English word
                    html += `<span class="word">${realTokens[i]}</span> `;
                } else {
                    // Masked: show jabberwocky word (styled as nonce)
                    html += `<span class="word nonce">${token}</span> `;
                }
            }

            html += `
                </div>
                <div class="controls">
                    <button class="guess-button" id="guess-btn">Make Guess</button>
                </div>
            `;

            return html;
        },
        choices: ['Make Guess'],
        button_html: '<button class="jspsych-btn" style="display:none;">%choice%</button>',
        on_load: function () {
            document.getElementById('guess-btn').addEventListener('click', function () {
                trialSequenceData.time_before_guess = Date.now() - startTime;
                jsPsych.finishTrial();
            });
        },
        trial_duration: null,
        response_ends_trial: false
    };
}

// ===== OPEN-ENDED TRIAL (Phase 2) =====
// Shows the full real passage and collects an open-ended response
// about what the participant thinks the passage is about.
// Response and timing are saved directly here (no separate guess/confidence screens).

function createOpenEndedTrial(trial, sectionTrialIndex, totalPhase2) {
    const passage = trial.jabber_passage || '';

    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: function () {
            startTime = Date.now();

            trialSequenceData = {
                subjCode:            subjCode,
                sublist:             sublistNumber,
                random_seed:         randomSeed,
                trial_type:          'open_ended',
                trial_number:        trial.trial_number,
                condition:           'open_ended',
                section_trial_index: sectionTrialIndex + 1,
                passage_id:          trial.passage_id,
                real_passage:        trial.real_passage || '',
                jabber_passage:      passage,
            };

            return `
                <div class="trial-counter">
                    Section 2 &mdash; Trial ${sectionTrialIndex + 1} of ${totalPhase2}
                </div>
                <div class="sentence-container open-ended-passage" id="sentence-container">
                    ${passage}
                </div>
                <div class="open-ended-question">
                    <label for="open-ended-response">
                        <strong>What do you think this passage is about?</strong>
                        Write a few sentences describing your interpretation.
                    </label>
                    <textarea
                        id="open-ended-response"
                        rows="5"
                        placeholder="Write your response here..."
                    ></textarea>
                </div>
                <div class="controls">
                    <button class="guess-button" id="submit-btn" disabled>Submit</button>
                </div>
            `;
        },
        choices: ['Submit'],
        button_html: '<button class="jspsych-btn" style="display:none;">%choice%</button>',
        on_load: function () {
            const textarea  = document.getElementById('open-ended-response');
            const submitBtn = document.getElementById('submit-btn');

            textarea.addEventListener('input', function () {
                submitBtn.disabled = textarea.value.trim().length === 0;
            });

            submitBtn.addEventListener('click', function () {
                trialSequenceData.open_ended_response = textarea.value.trim();
                trialSequenceData.time_before_submit  = Date.now() - startTime;
                consolidatedTrials.push({ ...trialSequenceData });
                console.log('Phase 2 trial saved:', trialSequenceData);
                jsPsych.finishTrial();
            });
        },
        trial_duration: null,
        response_ends_trial: false
    };
}

// ===== SHARED TRIAL TYPES =====

// Guess input — same for both sections
function createGuessInputTrial() {
    return {
        type: jsPsychSurveyText,
        questions: [{
            prompt: `
                <div class="instructions">
                    <p>What do you think the <strong>bolded word</strong> was in the sentence?</p>
                    <p><strong>Type ONE WORD for your guess:</strong></p>
                </div>
            `,
            name: 'target_word_guess',
            required: true,
            rows: 1,
            columns: 40
        }],
        on_finish: function (data) {
            trialSequenceData.guess    = data.response.target_word_guess;
            trialSequenceData.rt_guess = data.rt;
        }
    };
}

// Confidence rating — same for both sections.
// Pushes the completed trial object to consolidatedTrials.
function createConfidenceRatingTrial() {
    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: `
            <div style="text-align: center;">
                <p>How confident are you in your guess?</p>
            </div>
        `,
        choices: [
            'Not at all confident',
            'Slightly confident',
            'Moderately confident',
            'Very confident',
            'Extremely confident'
        ],
        on_finish: function (data) {
            trialSequenceData.confidence_rating = data.response + 1; // 0-4 → 1-5
            trialSequenceData.confidence_rt     = data.rt;
            consolidatedTrials.push({ ...trialSequenceData });
            console.log('Trial saved:', trialSequenceData);
        }
    };
}

// Feedback — shows the correct target word. Same for both sections.
function createFeedbackTrial(trial) {
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: function () {
            return `
                <div style="text-align: center; max-width: 600px; margin: 0 auto; padding: 40px;">
                    <h2>The target word was:</h2>
                    <p style="font-size: 36px; font-weight: bold; margin: 30px 0;">
                        ${trial.target_word}
                    </p>
                    <p style="font-size: 14px; color: #666;">
                        <em>Press any key to continue</em>
                    </p>
                </div>
            `;
        },
        trial_duration: null
    };
}

// ===== SCREENS =====

const consent = {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
        <div style="width: 800px; margin: 0 auto; text-align: left">
            <h3>Consent to Participate in Research</h3>

            <p>The task you are about to do is sponsored by University of Wisconsin-Madison.
            It is part of a protocol titled "What are we learning from language?"</p>

            <p>The task you are asked to do involves making simple responses to words and
            sentences. More detailed instructions for this specific task will be provided
            on the next screen.</p>

            <p>This task has no direct benefits. We do not anticipate any psychosocial
            risks. There is a risk of a confidentiality breach. Participants may become
            fatigued or frustrated due to the length of the study.</p>

            <p>The responses you submit as part of this task will be stored on a secure
            server and accessible only to researchers who have been approved by
            UW-Madison. Processed data with all identifiers removed could be used for
            future research studies or distributed to another investigator for future
            research studies without additional informed consent.</p>

            <p>You are free to decline to participate, to end participation at any time
            for any reason, or to refuse to answer any individual question without penalty
            or loss of earned compensation. We will not retain data from partial
            responses.</p>

            <p>If you have any questions or concerns about this task please contact the
            principal investigator: Prof. Gary Lupyan at lupyan@wisc.edu.</p>

            <p>If you are not satisfied with the response of the research team, have more
            questions, or want to talk with someone about your rights as a research
            participant, you should contact University of Wisconsin's Education Research
            and Social &amp; Behavioral Science IRB Office at 608-263-2320.</p>

            <p><strong>By clicking the box below, I consent to participate in this task
            and affirm that I am at least 18 years old.</strong></p>
        </div>
    `,
    choices: ['I Agree', 'I Do Not Agree'],
    on_finish: function (data) {
        if (data.response === 1) {
            jsPsych.endExperiment('Thank you for your time. The experiment has been ended.');
        }
    }
};

const welcome = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h1>Language Comprehension Experiment</h1>
            <p>In this experiment you will read passages and answer questions about them.
            The experiment has two parts. You will receive instructions for each part
            before it begins.</p>
            <p><em>Press any key to continue</em></p>
        </div>
    `
};

// --- Section 1 instructions ---

const baselineInstructions1 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Part 1 Instructions</h2>
            <p>In this part you will see passages where <strong>some words have been
            replaced with made-up nonsense words</strong>. The real words you can still
            see will give you context clues.</p>
            <p>Your job:</p>
            <ol>
                <li>Read the passage carefully.</li>
                <li>Try to figure out what the <strong>bolded word</strong> means.</li>
                <li>When you are ready to guess, click <strong>Make Guess</strong>.</li>
                <li>Type your best ONE-WORD guess.</li>
                <li>Rate your confidence.</li>
                <li>You will see the correct answer before moving on.</li>
            </ol>
            <p><em>Press any key to continue</em></p>
        </div>
    `
};

const baselineInstructions2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Part 1 — Tips</h2>
            <p>Some passages will have more nonsense words than others — that is intentional.</p>
            <p>Use every real word you can see as a clue. Even if you are unsure, make your
            best one-word guess before moving on.</p>
            <p><strong>Example:</strong></p>
            <p style="margin-left: 20px; font-style: italic;">
                "The glorp gleamed in the deng morning <strong>glosh</strong>."
            </p>
            <p>You might guess <em>sun</em>, <em>light</em>, or <em>air</em> — all
            reasonable one-word guesses based on context. That is the right level of
            specificity.</p>
            <p><strong>Please use ONE WORD guesses only.</strong></p>
            <p><em>Press any key to start Part 1</em></p>
        </div>
    `
};

// --- Transition between sections ---

const transitionScreen = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Great work — Part 1 complete!</h2>
            <p>Now we will move on to <strong>Part 2</strong>, which works differently.</p>
            <p><em>Press any key to read the Part 2 instructions</em></p>
        </div>
    `
};

// --- Section 2 instructions ---

const PHASE2_EXAMPLE_PASSAGE = [
    "Ghoc and splync . Splync he had gwob gwob . We 're doing a dwoque neight down .",
    "Knurt sneese to to to dwoque Neight Down dwazz down . It 's scis when you shroosh",
    "the throck wherg about this maunch in and out gheint . Knurt . The prerk wrudd is a",
    "dwazz . The prerk gheathe is a blalf which is fuite . Knurt . Why you wherg I did",
    "that ? I do n't ghegging greash because they twieve more sweil . Sneese to grong the",
    "dwazz dwazz but the twoofs off have scuthed about this phu plaiths the blalf Blaint",
    "in-and-out Gheint is the rirm in-and-out Gheint . Thweil , you greash what knime 's",
    "threrb a plause . Knurt , you greash sweil threrb plauses for gwal these drorbs Nalc",
    "gwalph whuile of the Dwazz-Dwazz . Knurt . No , just no do n't gwalph whuile of it",
    "crolt the shroosh crolt . The shroosh prerk wrudd is a blalf to is a dwazz . Knurt .",
    "That 's dwoll a flurl vewn flurl vewn dwaul threrb it Now threrb an grune shreight .",
    "That 's brulf . That 's sprate . We do n't we do n't twieve to yalt , you greash ,",
    "fru . So , knurt , it was brulf uzz . Knurt , it was girchs . So yipe plaith we did",
    "a dwoque splusk of dwoss . So this plaith we phleethed to do wrudd again and we did",
    "it for yisque yisque The Screrf and whadd cloop of dwoan strilges , which you will",
    "phiv why Thweil , we 've scuthed about it a thwipe whealt a cralph here. ",
].join(' ');

const phase2Instructions1 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 650px; margin: 0 auto; text-align: left;">
            <h2>Part 2 Instructions</h2>
            <p>In this part you will read passages where <strong>most words have been
            replaced with nonsense</strong>. Only common function words like
            <em>the</em>, <em>a</em>, <em>and</em>, <em>in</em>, etc. remain as real
            English words.</p>
            <p>After reading each passage you will answer:</p>
            <p style="margin: 16px 30px; font-size: 17px;">
                <em>"What do you think this passage is about?"</em>
            </p>
            <p>We know most of the words are unreadable — that is intentional. Use the
            function words, grammar, and overall structure to make your best guess at
            the meaning. Write a few sentences. There are no right or wrong answers.</p>
            <p><strong>Important:</strong> the nonsense words are randomly assigned
            and are not secretly similar to real words, so do not try to decode them
            letter by letter — focus on the structure of the text.</p>
            <p><em>Press any key to see examples</em></p>
        </div>
    `
};

const phase2Instructions2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 700px; margin: 0 auto; text-align: left;">
            <h2>Part 2 — Examples</h2>

            <p>Here is an example of what a passage will look like:</p>

            <div style="background: #fafafa; border: 1px solid #ddd; border-radius: 6px;
                        padding: 18px 22px; margin: 14px 0 24px 0;
                        font-size: 18px; line-height: 1.8;">
                ${PHASE2_EXAMPLE_PASSAGE}
            </div>

            <p>Below are examples of <strong>good</strong> and <strong>bad</strong>
            responses to that passage.</p>

            <div style="background: #f1f8f1; border-left: 4px solid #2e7d32;
                        padding: 14px 18px; margin: 14px 0; border-radius: 3px;">
                <p style="margin: 0 0 6px 0; font-weight: bold; color: #2e7d32;">
                    Good response
                </p>
                <p style="margin: 0;">
                    "The speaker is walking someone through a process or set of instructions, 
                    emphasizing how to perform actions correctly and what to watch out for."
                </p>
            </div>

            <div style="background: #f1f8f1; border-left: 4px solid #2e7d32;
                        padding: 14px 18px; margin: 14px 0; border-radius: 3px;">
                <p style="margin: 0 0 6px 0; font-weight: bold; color: #2e7d32;">
                    Good response
                </p>
                <p style="margin: 0;">
                    "A person talking to themselves about something they're trying 
                    to convince themselves they're sure of (they're not)."
                </p>
            </div>

            <div style="background: #fff4f4; border-left: 4px solid #c62828;
                        padding: 14px 18px; margin: 14px 0; border-radius: 3px;">
                <p style="margin: 0 0 6px 0; font-weight: bold; color: #c62828;">
                    Bad response — too vague, no attempt at interpretation
                </p>
                <p style="margin: 0; color: #555;">
                    "It's a story."
                </p>
            </div>

            <div style="background: #fff4f4; border-left: 4px solid #c62828;
                        padding: 14px 18px; margin: 14px 0; border-radius: 3px;">
                <p style="margin: 0 0 6px 0; font-weight: bold; color: #c62828;">
                    Bad response — too vague, no attempt at interpretation
                </p>
                <p style="margin: 0; color: #555;">
                    "It's a conversation."
                </p>
            </div>

            <p style="margin-top: 20px;">Do your best — even an uncertain interpretation
            is valuable to us.</p>
            <p><em>Press any key to start Part 2</em></p>
        </div>
    `
};

// ===== SAVING + END =====

const savingScreen = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="text-align: center; padding: 50px;">
            <h2>Saving your data...</h2>
            <p style="font-size: 18px; margin-top: 30px;">
                Please wait — do not close this window.
            </p>
            <div style="margin-top: 30px;">
                <div style="display: inline-block; width: 50px; height: 50px;
                     border: 5px solid #f3f3f3; border-top: 5px solid #2196f3;
                     border-radius: 50%; animation: spin 1s linear infinite;"></div>
            </div>
            <style>
                @keyframes spin {
                    0%   { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </div>
    `,
    choices: 'NO_KEYS',
    trial_duration: 1000
};

// ===== TIMELINE =====

async function createTimeline() {
    sublistNumber = await assignSublist();
    console.log(`SubjectCode: ${subjCode} | Sublist: ${sublistNumber} | Seed: ${randomSeed}`);
    await loadAllTrialData();

    const timeline = [
        consent,
        welcome,
        baselineInstructions1,
        baselineInstructions2
    ];

    // --- Section 1: baseline trials (some_masked + most_masked, shuffled) ---
    const totalBaseline = baselineTrialData.length;
    baselineTrialData.forEach((trial, i) => {
        timeline.push(createBaselineTrial(trial, i, totalBaseline));
        timeline.push(createGuessInputTrial());
        timeline.push(createConfidenceRatingTrial());
        timeline.push(createFeedbackTrial(trial));
    });

    // --- Transition ---
    timeline.push(transitionScreen);
    timeline.push(phase2Instructions1);
    timeline.push(phase2Instructions2);

    // --- Section 2: open-ended passage trials ---
    const totalPhase2 = phase2TrialData.length;
    phase2TrialData.forEach((trial, i) => {
        timeline.push(createOpenEndedTrial(trial, i, totalPhase2));
    });

    // --- Saving screen + data pipe save ---
    timeline.push(savingScreen);

    timeline.push({
        type: jsPsychPipe,
        action: 'save',
        experiment_id: 'PYSjeESL3lfq',
        filename: `${subjCode}.csv`,
        data_string: () => {
            console.log(`Saving ${consolidatedTrials.length} trials...`);
            if (consolidatedTrials.length > 0) {
                console.log('Columns:', Object.keys(consolidatedTrials[0]));
            }
            return arrayToCSV(consolidatedTrials);
        },
        on_finish: function (data) {
            if (data.success === false) {
                console.error('Data upload failed:', data);
            } else {
                console.log('Data upload successful.');
            }
        }
    });

    // --- Thank-you / redirect ---
    timeline.push({
        type: jsPsychHtmlKeyboardResponse,
        stimulus: function () {
            const surveyURL = getURLParameter('survey_url')
                || 'https://uwmadison.co1.qualtrics.com/jfe/form/SV_2gBjgNQpFFwXvhQ';
            const surveyWithId = `${surveyURL}${surveyURL.includes('?') ? '&' : '?'}subjCode=${subjCode}`;

            setTimeout(() => { window.location.href = surveyWithId; }, 2000);

            return `
                <div style="text-align: center; padding: 50px;">
                    <h2>Thank you!</h2>
                    <p style="font-size: 18px; margin: 30px 0;">
                        Your data has been saved successfully.
                    </p>
                    <p style="font-size: 16px; margin: 40px 0;">
                        You will be redirected to the final survey shortly...
                    </p>
                    <p style="font-size: 14px; color: #666; margin-top: 40px;">
                        If you are not redirected automatically,
                        <a href="${surveyWithId}" style="color: #2196f3;">click here</a>.
                    </p>
                </div>
            `;
        },
        choices: 'NO_KEYS',
        trial_duration: null
    });

    return timeline;
}

// ===== ENTRY POINT =====

createTimeline()
    .then(timeline => jsPsych.run(timeline))
    .catch(error => {
        console.error('Error loading experiment:', error);
        document.body.innerHTML = `
            <div style="text-align: center; padding: 50px;">
                <h2>Error Loading Experiment</h2>
                <p>Could not load <code>trial_lists/sublist_${sublistNumber}.csv</code>.</p>
                <p style="color: red;">Error: ${error.message}</p>
            </div>
        `;
    });
