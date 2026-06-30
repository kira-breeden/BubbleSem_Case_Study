// ============================================================
// BubbleSem v2 — Three-Section Experiment
//
// Section 1 (Baseline): some_masked + most_masked trials shuffled
//   - Target words shown with 20% or 40% of nonce words revealed
//   - No reveal interactivity; participant makes guess when ready
//
// Section 2 (Sampling): interactive click-to-reveal trials
//   - All nonce words are clickable; each reveal deducts points
//   - Participant reveals as few words as needed, then guesses
//
// Section 3 (Open-Ended): longer passages
//   - Participant reads each passage and answers an open-ended question
//     about what the passage is about
//   - Response and timing recorded; no target-word guessing
//
// CSV files required:
//   trial_lists/sublist_X.csv  — baseline + sampling trials (varies by sublist)
//     columns: condition, target_word, real_passage, jabber_passage,
//              target_word_position, unmasked_word_indices,
//              entropy, target_probability, trial_number, ...
//     condition values: some_masked | most_masked | sampling | open_ended
//
// URL parameters:
//   sublist=1..8  (default: 1)
//   subjCode=<string> (default: random ID)
// ============================================================

// ===== GLOBAL STATE =====

let baselineTrialData = [];
let samplingTrialData = [];
let phase2TrialData   = [];
let trialSequenceData = {};   // accumulates data across a trial's screens
let consolidatedTrials = [];  // all saved trial rows
let startTime = null;
let firstKeystrokeTime = null;  // time from passage appearing to first keypress in guess/response box

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

// Flash the points counter red briefly when points are deducted.
function updatePointsDisplay(points) {
    const el = document.getElementById('points-counter');
    if (!el) return;
    el.textContent = `Trial Points: ${Math.round(points)}`;
    el.style.color = '#d32f2f';
    setTimeout(() => { el.style.color = '#333'; }, 300);
}

// ===== CONDITION ASSIGNMENT =====

const EXPERIMENT_ID = 'HoczkSg9UdDc';
const N_SUBLISTS    = 8;

// Returns a sublist number (1–8).
// If ?sublist= is in the URL, uses that.
// Otherwise calls the DataPipe condition assignment API for counterbalanced assignment.
async function assignSublist() {
    const param = getURLParameter('sublist');
    if (param) {
        console.log(`Sublist from URL: ${param}`);
        return param;
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
    const allTrials = await loadCSV(`trial_lists/${sublistNumber}.csv`);

    const baseline  = allTrials.filter(t => t.condition === 'some_masked' || t.condition === 'most_masked');
    const sampling  = allTrials.filter(t => t.condition === 'sampling');
    const openEnded = allTrials.filter(t => t.condition === 'open_ended');

    // Shuffle each section independently with seeded RNG
    const rng = new SeededRandom(randomSeed);
    baselineTrialData = rng.shuffle(baseline);
    samplingTrialData = rng.shuffle(sampling);
    phase2TrialData   = rng.shuffle(openEnded);

    console.log(`Baseline trials: ${baselineTrialData.length}`);
    console.log(`Sampling trials: ${samplingTrialData.length}`);
    console.log(`Open-ended trials: ${phase2TrialData.length}`);
}

// ===== HARDCODED PRACTICE + ATTENTION CHECK TRIALS =====

const PRACTICE_TRIAL_DATA = [
    {
        passageHtml: `The zirps kicked the <span class="word target">blorf</span> across the scrempf. It glashed high prof the deek.`,
        targetWord:  'ball',
    },
    {
        passageHtml: `She zop dake at the glimp and opened her <span class="word target">glorp</span>. She began to zap quietly.`,
        targetWord:  'book',
    },
];

// Same passages as PRACTICE_TRIAL_DATA but fully masked with nonce words for
// the Section 2 practice, where participants click to reveal words.
const PRACTICE_SAMPLING_DATA = [
    {
        jabber_passage:       'The zirps flirped the blorf plomp the scrempf. Zelf glashed grop prof the deek.',
        real_passage:         'The kids kicked the ball across the field. It soared high over the fence.',
        target_word_position: 4,
        targetWord:           'ball',
    },
    {
        jabber_passage:       'Thrix zop dake at the glimp and grunted gliv glorp. Thrix whaped to zap druply.',
        real_passage:         'She sat down at the table and opened her book. She began to read quietly.',
        target_word_position: 9,
        targetWord:           'book',
    },
];

const ATTENTION_CHECK_DATA = [
    {
        passageHtml: `The dog barked loudly at the strange <span class="word target">blorf</span> across the street. Everyone on the block could hear it.`,
        targetWord: 'cat',
    },
    {
        passageHtml: `She turned on the kitchen <span class="word target">glorp</span> to fill the pot with water. The sound of the cool tap water filling the pot echoed aorund the kitchen.`,
        targetWord: 'light',
    },
];

function createHardcodedTrial(passageHtml, targetWord, trialType, trialNumber) {
    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: function () {
            startTime = Date.now();
            firstKeystrokeTime = null;
            trialSequenceData = {
                subjCode,
                sublist:     sublistNumber,
                random_seed: randomSeed,
                trial_type:  trialType,
                trial_number: trialNumber,
                target_word:  targetWord,
                condition:    trialType,
            };
            return `
                <div class="sentence-container baseline-passage" id="sentence-container">
                    ${passageHtml}
                </div>
                <div class="controls">
                    <button class="guess-button" id="guess-btn">Make a Guess</button>
                </div>
            `;
        },
        choices: ['Make a Guess'],
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

// ===== HARDCODED SAMPLING TRIAL =====
// Same passages as the baseline practice trials, but rendered in sampling style:
// all nonce words are clickable, revealing the real English word on click.

function createHardcodedSamplingTrial(p, trialType, trialNumber) {
    const jabberTokens   = tokenizeSentence(p.jabber_passage);
    const realTokens     = tokenizeSentence(p.real_passage);
    const targetTokenIdx = wordPosToTokenIndex(jabberTokens, p.target_word_position);

    const numRevealableWords = getMaskableTokenIndices(jabberTokens, realTokens, targetTokenIdx).length;
    const pointsPerReveal    = numRevealableWords > 0
        ? Math.round((100 / numRevealableWords) * 100) / 100
        : 0;

    let trialPoints   = 100;
    let revealedWords = new Set();

    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: function () {
            startTime     = Date.now();
            trialPoints   = 100;
            revealedWords = new Set();

            trialSequenceData = {
                subjCode,
                sublist:      sublistNumber,
                random_seed:  randomSeed,
                trial_type:   trialType,
                trial_number: trialNumber,
                target_word:  p.targetWord,
                condition:    trialType,
            };

            let html = `
                <div class="points-counter" id="points-counter">Trial Points: ${trialPoints}</div>
                <div class="sentence-container sampling-passage" id="sentence-container">
            `;

            for (let i = 0; i < jabberTokens.length; i++) {
                const token = jabberTokens[i];
                const nextIsPunct = (i + 1 < jabberTokens.length) && isPunct(jabberTokens[i + 1]);
                const sep = nextIsPunct ? '' : ' ';

                if (isPunct(token)) {
                    html += token;
                    if (/[.,!?;:]/.test(token) && i < jabberTokens.length - 1) html += ' ';
                    continue;
                }

                if (i === targetTokenIdx) {
                    html += `<span class="word target">${token}</span>${sep}`;
                } else if (isAutoRevealed(jabberTokens[i], realTokens[i])) {
                    html += `<span class="word article">${realTokens[i]}</span>${sep}`;
                } else {
                    html += `<span class="word clickable" data-index="${i}">${token}</span>${sep}`;
                }
            }

            html += `
                </div>
                <div class="controls">
                    <button class="guess-button" id="guess-btn">Make a Guess</button>
                </div>
            `;

            return html;
        },
        choices: ['Make a Guess'],
        button_html: '<button class="jspsych-btn" style="display:none;">%choice%</button>',
        on_load: function () {
            document.querySelectorAll('.sampling-passage .word.clickable').forEach(wordEl => {
                wordEl.addEventListener('click', function () {
                    const index = parseInt(this.dataset.index);
                    if (!revealedWords.has(index)) {
                        revealedWords.add(index);
                        trialPoints = Math.max(0, trialPoints - pointsPerReveal);
                        updatePointsDisplay(trialPoints);
                        this.textContent = realTokens[index];
                        this.classList.remove('clickable');
                        this.classList.add('revealed');
                    }
                });
            });

            document.getElementById('guess-btn').addEventListener('click', function () {
                trialSequenceData.time_before_guess = Date.now() - startTime;
                jsPsych.finishTrial();
            });
        },
        trial_duration: null,
        response_ends_trial: false
    };
}

// ===== BASELINE TRIAL =====
// Shows the passage with predetermined masked/unmasked words.
// No reveal interactivity — participant clicks "Make a Guess" when ready.

function createBaselineTrial(trial, sectionTrialIndex, totalBaseline, trialNumber) {
    const realSentence   = trial.real_passage    || '';
    const jabberSentence = trial.jabber_passage  || '';
    const realTokens     = tokenizeSentence(realSentence);
    const jabberTokens   = tokenizeSentence(jabberSentence);
    // Use jabberTokens for targetTokenIdx — the render loop iterates jabberTokens,
    // so the index must come from the same tokenization.
    const targetTokenIdx = wordPosToTokenIndex(jabberTokens, trial.target_word_position);
    const maskingLevel   = trial.condition || 'some_masked';

    // Revealed words come entirely from the predetermined list in the CSV.
    const unmaskedWordPositions = parseJSONColumn(trial.unmasked_word_indices);
    const unmaskedTokenIdxSet = new Set(
        unmaskedWordPositions
            .map(pos => wordPosToTokenIndex(jabberTokens, pos))
            .filter(i => i >= 0)
    );

    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: function () {
            startTime = Date.now();

            trialSequenceData = {
                subjCode:               subjCode,
                sublist:                sublistNumber,
                random_seed:            randomSeed,
                trial_type:             'baseline',
                trial_number:           trialNumber,
                trial_list_index:       trial.trial_number,
                condition:              maskingLevel,
                section_trial_index:    sectionTrialIndex + 1,
                target_word:            trial.target_word,
                target_word_position:   trial.target_word_position,
                entropy:                trial.entropy,
                target_probability:     trial.target_probability,
                real_passage:           realSentence,
                jabber_passage:         jabberSentence,
                unmasked_word_indices: JSON.stringify(unmaskedWordPositions),
            };

            let html = `
                <div class="sentence-container baseline-passage" id="sentence-container">
            `;

            for (let i = 0; i < jabberTokens.length; i++) {
                const token = jabberTokens[i];

                if (isPunct(token)) {
                    html += token;
                    if (/[.,!?;:]/.test(token) && i < jabberTokens.length - 1) html += ' ';
                    continue;
                }

                const nextIsPunct = (i + 1 < jabberTokens.length) && isPunct(jabberTokens[i + 1]);
                const sep = nextIsPunct ? '' : ' ';

                if (i === targetTokenIdx) {
                    // Target word: always show as jabberwocky, bold
                    html += `<span class="word target">${token}</span>${sep}`;
                } else if (
                    unmaskedTokenIdxSet.has(i) ||
                    isAutoRevealed(jabberTokens[i], realTokens[i])
                ) {
                    // Unmasked: show real English word
                    html += `<span class="word">${realTokens[i]}</span>${sep}`;
                } else {
                    // Masked: show jabberwocky word (styled as nonce)
                    html += `<span class="word nonce">${token}</span>${sep}`;
                }
            }

            html += `
                </div>
                <div class="controls">
                    <button class="guess-button" id="guess-btn">Make a Guess</button>
                </div>
            `;

            return html;
        },
        choices: ['Make a Guess'],
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

// ===== SAMPLING TRIAL (Section 2) =====
// Shows the passage with all nonce words as clickable buttons.
// Each click reveals the real word and deducts points.
// Participant clicks "Make a Guess" when ready, then guess/confidence/feedback follow.

function createSamplingTrial(trial, sectionTrialIndex, trialNumber) {
    const realSentence   = trial.real_passage   || '';
    const jabberSentence = trial.jabber_passage || '';
    const realTokens     = tokenizeSentence(realSentence);
    const jabberTokens   = tokenizeSentence(jabberSentence);
    const targetTokenIdx = wordPosToTokenIndex(jabberTokens, trial.target_word_position);

    const numRevealableWords = getMaskableTokenIndices(jabberTokens, realTokens, targetTokenIdx).length;
    const pointsPerReveal    = numRevealableWords > 0
        ? Math.round((100 / numRevealableWords) * 100) / 100
        : 0;
    const tokenToWordPos = buildTokenToWordPosMap(jabberTokens);

    // Mutable state shared between stimulus() and on_load() via closure.
    let trialPoints   = 100;
    let revealedWords = new Set();
    let clickTimes    = [];

    return {
        type: jsPsychHtmlButtonResponse,
        stimulus: function () {
            startTime     = Date.now();
            trialPoints   = 100;
            revealedWords = new Set();
            clickTimes    = [];

            trialSequenceData = {
                subjCode,
                sublist:              sublistNumber,
                random_seed:          randomSeed,
                trial_type:           'sampling',
                trial_number:         trialNumber,
                trial_list_index:     trial.trial_number,
                condition:            'sampling',
                section_trial_index:  sectionTrialIndex + 1,
                target_word:          trial.target_word,
                target_word_position: trial.target_word_position,
                entropy:              trial.entropy,
                target_probability:   trial.target_probability,
                real_passage:         realSentence,
                jabber_passage:       jabberSentence,
                num_revealable_words: numRevealableWords,
                points_per_reveal:    pointsPerReveal,
            };

            let html = `
                <div class="points-counter" id="points-counter">Trial Points: ${trialPoints}</div>
                <div class="sentence-container sampling-passage" id="sentence-container">
            `;

            for (let i = 0; i < jabberTokens.length; i++) {
                const token = jabberTokens[i];
                const nextIsPunct = (i + 1 < jabberTokens.length) && isPunct(jabberTokens[i + 1]);
                const sep = nextIsPunct ? '' : ' ';

                if (isPunct(token)) {
                    html += token;
                    if (/[.,!?;:]/.test(token) && i < jabberTokens.length - 1) html += ' ';
                    continue;
                }

                if (i === targetTokenIdx) {
                    html += `<span class="word target">${token}</span>${sep}`;
                } else if (isAutoRevealed(jabberTokens[i], realTokens[i])) {
                    html += `<span class="word article">${realTokens[i]}</span>${sep}`;
                } else {
                    html += `<span class="word clickable" data-index="${i}">${token}</span>${sep}`;
                }
            }

            html += `
                </div>
                <div class="controls">
                    <button class="guess-button" id="guess-btn">Make a Guess</button>
                </div>
            `;

            return html;
        },
        choices: ['Make a Guess'],
        button_html: '<button class="jspsych-btn" style="display:none;">%choice%</button>',
        on_load: function () {
            document.querySelectorAll('.sampling-passage .word.clickable').forEach(wordEl => {
                wordEl.addEventListener('click', function () {
                    const index = parseInt(this.dataset.index);
                    if (!revealedWords.has(index)) {
                        revealedWords.add(index);
                        trialPoints = Math.max(0, trialPoints - pointsPerReveal);
                        updatePointsDisplay(trialPoints);
                        clickTimes.push({
                            word_index:      index,
                            revealed_word:   realTokens[index],
                            time_from_start: Date.now() - startTime
                        });
                        this.textContent = realTokens[index];
                        this.classList.remove('clickable');
                        this.classList.add('revealed');
                    }
                });
            });

            document.getElementById('guess-btn').addEventListener('click', function () {
                const revealedWordsList = Array.from(revealedWords)
                    .filter(idx => !isPunct(realTokens[idx]))
                    .map(idx => realTokens[idx]);

                const unmaskedWordPositions = Array.from(revealedWords)
                    .filter(idx => !isPunct(realTokens[idx]))
                    .map(idx => tokenToWordPos.get(idx))
                    .filter(pos => pos !== undefined)
                    .sort((a, b) => a - b);

                trialSequenceData.unmasked_word_indices   = JSON.stringify(unmaskedWordPositions);
                trialSequenceData.num_words_revealed      = revealedWordsList.length;
                trialSequenceData.revealed_words          = JSON.stringify(revealedWordsList);
                trialSequenceData.click_times             = JSON.stringify(clickTimes);
                trialSequenceData.total_time_before_guess = Date.now() - startTime;
                trialSequenceData.points_remaining        = Math.round(trialPoints * 100) / 100;

                jsPsych.finishTrial();
            });
        },
        trial_duration: null,
        response_ends_trial: false
    };
}

// ===== OPEN-ENDED TRIAL (Section 3) =====
// Shows the full real passage and collects an open-ended response
// about what the participant thinks the passage is about.
// Response and timing are saved directly here (no separate guess/confidence screens).

function createOpenEndedTrial(trial, sectionTrialIndex, totalPhase2, trialNumber) {
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
                trial_number:        trialNumber,
                trial_list_index:    trial.trial_number,
                condition:           'open_ended',
                section_trial_index: sectionTrialIndex + 1,
                passage_id:          trial.passage_id,
                real_passage:        trial.real_passage || '',
                jabber_passage:      passage,
            };

            return `
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
            firstKeystrokeTime = null;
            const textarea  = document.getElementById('open-ended-response');
            const submitBtn = document.getElementById('submit-btn');

            textarea.addEventListener('keydown', function () {
                if (firstKeystrokeTime === null) {
                    firstKeystrokeTime = Date.now() - startTime;
                }
            }, { once: true });

            textarea.addEventListener('input', function () {
                submitBtn.disabled = textarea.value.trim().length === 0;
            });

            submitBtn.addEventListener('click', function () {
                trialSequenceData.open_ended_response     = textarea.value.trim();
                trialSequenceData.time_before_submit      = Date.now() - startTime;
                trialSequenceData.time_to_first_keystroke = firstKeystrokeTime;
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
        on_load: function () {
            firstKeystrokeTime = null;
            const input     = document.querySelector('[data-name="target_word_guess"]');
            const submitBtn = document.querySelector('input[type="submit"].jspsych-btn');

            if (submitBtn) submitBtn.disabled = true;

            if (input) {
                input.addEventListener('keydown', function () {
                    if (firstKeystrokeTime === null) {
                        firstKeystrokeTime = Date.now() - startTime;
                    }
                }, { once: true });

                input.addEventListener('input', function () {
                    if (submitBtn) submitBtn.disabled = input.value.trim().length === 0;
                });
            }
        },
        on_finish: function (data) {
            trialSequenceData.guess                   = data.response.target_word_guess;
            trialSequenceData.rt_guess                = data.rt;
            trialSequenceData.time_to_first_keystroke = firstKeystrokeTime;
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

// Feedback — shows the participant's guess alongside the correct target word.
function createFeedbackTrial(trial) {
    return {
        type: jsPsychHtmlKeyboardResponse,
        stimulus: function () {
            const participantGuess = (trialSequenceData.guess || '').trim();
            const correct = trial.target_word;
            return `
                <div style="text-align: center; max-width: 2000px; margin: 0 auto; padding: 40px;">
                    <h2 style="margin-bottom: 30px;">Good job!</h2>
                    <div style="display: flex; justify-content: center; gap: 60px;">
                        <div style="flex: 1; min-width: 220px; background: #f0f0f0; border-radius: 8px; padding: 20px;">
                            <p style="font-size: 13px; color: #555; margin: 0 0 10px;">Your response</p>
                            <p style="font-size: 30px; font-weight: bold; margin: 0;">${participantGuess || '—'}</p>
                        </div>
                        <div style="flex: 1; min-width: 220px; background: #f0f0f0; border-radius: 8px; padding: 20px;">
                            <p style="font-size: 13px; color: #555; margin: 0 0 10px;">Correct response</p>
                            <p style="font-size: 30px; font-weight: bold; margin: 0;">${correct}</p>
                        </div>
                    </div>
                    <p style="font-size: 14px; color: #666; margin-top: 30px;">
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
            <h1>Welcome!</h1>
            <p>In this experiment you will read passages and answer questions about them.
            The experiment has three parts. You will receive instructions for each part
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
            <p>In this task you will read a passage of text and try to guess the meaning of a word in bold. The majority of 
            the words in the passage will be nonsense words. These words are totally random and have no relationship to the 
            real English words they have replaced. </p>
            <p>On each trial:</p>
            <ol>
                <li>You'll see a sentence with one <strong>bolded word</strong> - this is your target word to guess</li>
                <li>Read the sentence carefully to understand the context</li>
                <li>When you think you know the meaning of the bolded word, KEEP YOUR GUESS IN MIND and click "Make a Guess"</li>
                <li>Type your guess for the bolded word</li>
                <li>Rate your confidence in your guess</li>
                <li>You'll see feedback showing the correct answer</li>
            </ol>
            <p><strong>Important: Try to be as specific as possible in your guesses. Your guess should be ONE WORD!</strong></p>
            <p><em>Press any key to move on to the next page </em></p>
        </div> 
    `
};

const baselineInstructions2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>When are you ready to guess?</h2>
            <p><strong>This will sometimes be quite difficult, but just do your best!</strong></p>
            <p>For example, you might see something like this:</p>
            <p style="margin-left: 20px; font-style: italic;">
                "The glorp tafed in the deng zirp <strong>glosh</strong>."
            </p>
            <p>Take some time to think about what "glosh" might mean and once you have your best ONE WORD GUESS, keep that guess in mind and click "Make a Guess".</p>
            <p style="margin-top: 30px;"><em>Press any key to continue</em></p>
        </div>
    `
};

// --- Practice instructions ---

const practiceInstructions1 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Let's Practice</h2>
            <p>Before we begin, let's do <strong>2 practice trials</strong> so you can
            get comfortable with the task.</p>
            <p>Remember: the bolded nonsense word is what you're guessing. Some words will be masked
            and some won't but try your best to guess the meaning.</p>
            <p><em>Press any key to start the practice</em></p>
        </div>
    `
};

// --- Transition 1 → 2 ---

const practiceCompleteScreen = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Practice complete!</h2>
            <p><strong>Final Reminder: Please use ONE WORD GUESSES!</strong></p>
            <p><strong>Some will be harder than others, so just do your best and take your time!</strong></p>
            <p><em>Press any key to start Part 1</em></p>
        </div>
    `
};

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

// --- Section 2 (Sampling) instructions ---

const samplingInstructions1 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Part 2 Instructions</h2>
            <p>In this part you will again see sentences with made-up nonsense words.</p>
            <p>The difference: now you can <strong>click on any nonsense word to reveal
            its real meaning</strong>.</p>
            <p>Your job:</p>
            <ol>
                <li>Read the sentence.</li>
                <li>Click words to reveal them if you need more context — but <strong> try to
                    reveal as few as possible! </strong> </li>
                <li>When you are ready to guess, click <strong>Make a Guess</strong>.</li>
                <li>Type your best ONE-WORD guess for the <strong>bolded word</strong>.</li>
                <li>Rate your confidence.</li>
                <li>You will see the correct answer before moving on.</li>
            </ol>
            <p><strong>Scoring:</strong> Each trial starts with <strong>100 points</strong>.
            Each word you reveal costs points. Try to guess with as few reveals as
            possible!</p>
            <p><em>Press any key to continue</em></p>
        </div>
    `
};

const samplingInstructions2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>When are you ready to guess?</h2>
            
            <p><strong>Do not guess the word if you have no idea what it might mean.</strong></p>
            
            <p>For example:</p>
            
            <p style="margin-left: 20px; font-style: italic;">
                "The glorp tafed in the deng zirp <strong>glosh</strong>."
            </p>
            
            <p>You might think that the target word is an object, but this is not a specific enough guess.</p>
            
            <p>Let's reveal some more words!</p>
            
            <p style="margin-left: 20px; font-style: italic;">
                "The glorp gleamed in the deng morning <strong>glosh</strong>."
            </p>
            
            <p>Now you might have some better guesses about what <strong>glosh</strong> could be! Is it maybe sun? sunshine? air? light? </p>
            <p> <strong>This is the right level of specificity for your guess. </strong></p>
            
            <p style="margin-top: 30px;"><em>Press any key to continue</em></p>
        </div>
    `
};

const samplingInstructions3 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">

            <p>However! You might not be able to get this close every time:<p>
                        
            <p>Sometimes, your best guess might just be that it's an animal, a color, a type of plant, etc. These are okay guesses, though they are not as good as the earlier ones.</p>

            <p>You should be able to narrow down the meaning more than just what part of speech it might be, or that it might be an object that moves. </p>
            
            <p><strong>Try and get as close as you can without losing too many points. We are looking for one word answers. </strong> </p>
            
            <p style="margin-top: 30px;"><em>Press any key to begin Part 2 practice.</em></p>
        </div>
    `
};

const practiceInstructions2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Let's Practice</h2>
            <p>Before we begin, let's do <strong>2 practice trials</strong> so you can
            get comfortable with the task.</p>
            <p>Remember: Click on words to reveal their meaning, but try to reveal as few as possible! </p>
            <p><em>Press any key to start the practice</em></p>
        </div>
    `
};

// --- Transition 2 → 3 ---

const practiceCompleteScreen2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Practice complete!</h2>
            <p><strong>Final Reminder: Please use ONE WORD GUESSES!</strong></p>
            <p><strong>Just do your best and take your time!</strong></p>
            <p><em>Press any key to start Part 2</em></p>
        </div>
    `
};

const transitionScreen2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 600px; margin: 0 auto; text-align: left;">
            <h2>Great work — Part 2 complete!</h2>
            <p>Now we will move on to <strong>Part 3</strong>, which works differently.</p>
            <p><em>Press any key to read the Part 3 instructions</em></p>
        </div>
    `
};

// --- Section 3 (Open-Ended) instructions ---

const PHASE2_EXAMPLE_PASSAGE = [
    "Ghoc and splync. Splync he had gwob gwob. We're doing a dwoque neight down.",
    "Knurt sneese to to to dwoque Neight Down dwazz down. It's scis when you shroosh",
    "the throck wherg about this maunch in and out gheint. Knurt. The prerk wrudd is a",
    "dwazz. The prerk gheathe is a blalf which is fuite. Knurt. Why you wherg I did",
    "that? I don't ghegging greash because they twieve more sweil. Sneese to grong the",
    "dwazz dwazz but the twoofs off have scuthed about this phu plaiths the blalf Blaint",
    "in-and-out Gheint is the rirm in-and-out Gheint. Thweil, you greash what knime's",
    "threrb a plause. Knurt, you greash sweil threrb plauses for gwal these drorbs Nalc",
    "gwalph whuile of the Dwazz-Dwazz. Knurt. No, just no don't gwalph whuile of it",
    "crolt the shroosh crolt . The shroosh prerk wrudd is a blalf to is a dwazz. Knurt.",
    "That's dwoll a flurl vewn flurl vewn dwaul threrb it Now threrb an grune shreight.",
    "That's brulf. That's sprate. We don't we don't twieve to yalt, you greash,",
    "fru. So, knurt, it was brulf uzz. Knurt, it was girchs. So yipe plaith we did",
    "a dwoque splusk of dwoss. So this plaith we phleethed to do wrudd again and we did",
    "it for yisque yisque The Screrf and whadd cloop of dwoan strilges, which you will",
    "phiv why Thweil, we've scuthed about it a thwipe whealt a cralph here.",
].join(' ');

const phase2Instructions1 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 650px; margin: 0 auto; text-align: left;">
            <h2>Part 3 Instructions</h2>
            <p>In this part you will read longer passages where <strong>most words have been
            replaced with nonsense</strong>. 
            <p>After reading each passage you will answer:</p>
            <p style="margin: 16px 30px; font-size: 17px;">
                <em>"What do you think this passage is about?"</em>
            </p>
            <p>We know this might seem pretty difficult with most of words masked with nonsense. 
            But, there are no right or wrong answers! Do your best to understand and make a guess 
            about what the passage is about.</p>
            <p><strong>Important:</strong> the nonsense words are randomly assigned
            and are not secretly related to the real words.</p>
            <p><em>Press any key to see examples</em></p>
        </div>
    `
};

const phase2Instructions2 = {
    type: jsPsychHtmlKeyboardResponse,
    stimulus: `
        <div style="max-width: 700px; margin: 0 auto; text-align: left;">
            <h2>Part 3 — Examples</h2>

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
                    "A person talking to someone else on a podcast. They are talking 
                    about their beliefs and morals, agreeing that they do not partake in 
                    certain things."
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
            <p><em>Press any key to start Part 3</em></p>
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
        baselineInstructions2,
        practiceInstructions1,
    ];

    // --- Practice trials ---
    PRACTICE_TRIAL_DATA.forEach((p, i) => {
        timeline.push(createHardcodedTrial(p.passageHtml, p.targetWord, 'practice', `P${i + 1}`));
        timeline.push(createGuessInputTrial());
        timeline.push(createConfidenceRatingTrial());
        timeline.push(createFeedbackTrial({ target_word: p.targetWord }));
    });

    timeline.push(practiceCompleteScreen);

    // --- Section 1: baseline trials with attention checks at ~1/3 and ~2/3 ---
    const totalBaseline = baselineTrialData.length;
    let globalTrialNum = 1;
    let attnCheckIdx = 0;
    const attnInsertAfter = new Set([
        Math.floor(totalBaseline / 3) - 1,
        Math.floor(2 * totalBaseline / 3) - 1,
    ]);

    baselineTrialData.forEach((trial, i) => {
        timeline.push(createBaselineTrial(trial, i, totalBaseline, globalTrialNum++));
        timeline.push(createGuessInputTrial());
        timeline.push(createConfidenceRatingTrial());
        timeline.push(createFeedbackTrial(trial));

        if (attnInsertAfter.has(i) && attnCheckIdx < ATTENTION_CHECK_DATA.length) {
            const check = ATTENTION_CHECK_DATA[attnCheckIdx];
            timeline.push(createHardcodedTrial(check.passageHtml, check.targetWord, 'attention_check', `AC${attnCheckIdx + 1}`));
            timeline.push(createGuessInputTrial());
            timeline.push(createConfidenceRatingTrial());
            timeline.push(createFeedbackTrial({ target_word: check.targetWord }));
            attnCheckIdx++;
        }
    });

    // --- Transition 1 → 2 ---
    timeline.push(transitionScreen);
    timeline.push(samplingInstructions1);
    timeline.push(samplingInstructions2);
    timeline.push(samplingInstructions3);
    timeline.push(practiceInstructions2);

    // --- Section 2 practice trials ---
    PRACTICE_SAMPLING_DATA.forEach((p, i) => {
        timeline.push(createHardcodedSamplingTrial(p, 'practice_s2', `P2_${i + 1}`));
        timeline.push(createGuessInputTrial());
        timeline.push(createConfidenceRatingTrial());
        timeline.push(createFeedbackTrial({ target_word: p.targetWord }));
    });

    timeline.push(practiceCompleteScreen2);

    // --- Section 2: sampling trials (interactive click-to-reveal) ---
    samplingTrialData.forEach((trial, i) => {
        timeline.push(createSamplingTrial(trial, i, globalTrialNum++));
        timeline.push(createGuessInputTrial());
        timeline.push(createConfidenceRatingTrial());
        timeline.push(createFeedbackTrial(trial));
    });

    // --- Transition 2 → 3 ---
    timeline.push(transitionScreen2);
    timeline.push(phase2Instructions1);
    timeline.push(phase2Instructions2);

    // --- Section 3: open-ended passage trials ---
    const totalPhase2 = phase2TrialData.length;
    phase2TrialData.forEach((trial, i) => {
        timeline.push(createOpenEndedTrial(trial, i, totalPhase2, globalTrialNum++));
    });

    // --- Saving screen + data pipe save ---
    timeline.push(savingScreen);

    timeline.push({
        type: jsPsychPipe,
        action: 'save',
        experiment_id: EXPERIMENT_ID,
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
                || 'https://uwmadison.co1.qualtrics.com/jfe/form/SV_2aRxcJmUkeVr06W';
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
                <p>Could not load <code>trial_lists/${sublistNumber}.csv</code>.</p>
                <p style="color: red;">Error: ${error.message}</p>
            </div>
        `;
    });
