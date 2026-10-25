const datasetCountEl = document.getElementById('dataset-count');
const searchForm = document.getElementById('song-search-form');
const searchFeedback = document.getElementById('search-feedback');
const searchResults = document.getElementById('search-results');
const playlistForm = document.getElementById('playlist-form');
const playlistFeedback = document.getElementById('playlist-feedback');
const playlistSummary = document.getElementById('playlist-summary');
const playlistResults = document.getElementById('playlist-results');
const moodResults = document.getElementById('mood-results');
const modelInfo = document.getElementById('model-info');

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || 'Something went wrong.');
  }
  return data;
}

function setFeedback(element, message, kind = '') {
  element.textContent = message;
  element.className = 'feedback';
  if (kind) {
    element.classList.add(kind);
  }
}

function emptyState(message) {
  return `<div class="empty-state">${message}</div>`;
}

function renderSongCard(song) {
  const audioFeatures = song.audio_features || {};
  const featurePills = Object.entries(audioFeatures)
    .slice(0, 4)
    .map(([key, value]) => `<span class="pill">${key}: ${Number(value).toFixed(2)}</span>`)
    .join('');

  const artwork = song.artwork_url
    ? `<img src="${song.artwork_url}" alt="${song.track_name}" style="width:100%; height:110px; object-fit:cover; border-radius:12px; border:1px solid rgba(255,255,255,0.08); margin-bottom:12px;" />`
    : '<div class="artwork-placeholder" aria-label="No artwork available"><span aria-hidden="true">♪</span><small>No artwork</small></div>';

  return `
    <article class="song-card">
      ${artwork}
      <div class="song-meta">
        <span class="badge">${song.mood || 'Mood pending'}</span>
      </div>
      <h3>${song.track_name || 'Unknown track'}</h3>
      <p class="reason">${song.artist_name || 'Unknown artist'}</p>
      <div class="feature-list">${featurePills || '<span class="pill">Audio features unavailable</span>'}</div>
      <p class="reason">${song.mood_reason || 'Audio features are being compared against the mood rules.'}</p>
      <button class="select-button" type="button" data-song='${JSON.stringify(song).replace(/'/g, '&apos;')}'>Predict mood</button>
    </article>
  `;
}

function renderPlaylistCard(item) {
  const features = Object.entries(item.audio_features || {})
    .slice(0, 4)
    .map(([key, value]) => `<span class="pill">${key}: ${Number(value).toFixed(2)}</span>`)
    .join('');

  return `
    <article class="playlist-card">
      <div class="song-meta">
        <span class="badge">${item.predicted_mood}</span>
      </div>
      <h3>${item.track_name}</h3>
      <p class="reason">${item.artist_name}</p>
      <div class="feature-list">${features || '<span class="pill">No matching features</span>'}</div>
      <p class="reason">${item.mood_reason}</p>
    </article>
  `;
}

async function loadDatasetCount() {
  try {
    const data = await fetchJson('/api/health');
    datasetCountEl.textContent = `${Number(data.dataset_rows || 0).toLocaleString()} songs`;
  } catch (error) {
    datasetCountEl.textContent = 'Dataset unavailable';
  }
}

async function handleSongSearch(event) {
  event.preventDefault();
  const title = document.getElementById('search-title').value.trim();
  const artist = document.getElementById('search-artist').value.trim();

  if (!title && !artist) {
    setFeedback(searchFeedback, 'Please enter a song title or artist before searching.', 'error');
    return;
  }

  setFeedback(searchFeedback, 'Searching the local dataset…');
  searchResults.innerHTML = '';

  try {
    const data = await fetchJson(`/api/search?q=${encodeURIComponent(title)}&artist=${encodeURIComponent(artist)}`);
    if (!data.results || data.results.length === 0) {
      setFeedback(searchFeedback, data.message || 'No matching songs were found in the local dataset. Try a different title or artist.', 'error');
      searchResults.innerHTML = emptyState('No matching songs found in the local dataset.');
      return;
    }

    setFeedback(searchFeedback, `${data.results.length} matching song(s) found. Select one to predict the mood.`, 'success');
    searchResults.innerHTML = data.results.map(renderSongCard).join('');

    document.querySelectorAll('[data-song]').forEach((button) => {
      button.addEventListener('click', async () => {
        const song = JSON.parse(button.getAttribute('data-song'));
        await predictSong(song);
      });
    });
  } catch (error) {
    setFeedback(searchFeedback, error.message, 'error');
    searchResults.innerHTML = emptyState('The local dataset search failed. Please try again.');
  }
}

async function predictSong(song) {
  const payload = {
    song: {
      track_name: song.track_name,
      artist_name: song.artist_name,
      album_name: song.album_name,
      language: song.language,
      year: song.year,
    },
    features: song.audio_features || {},
  };

  try {
    const data = await fetchJson('/api/predict-mood', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const card = document.createElement('article');
    card.className = 'song-card';
    card.innerHTML = `
      <div class="song-meta"><span class="badge">${data.predicted_mood}</span></div>
      <h3>${song.track_name}</h3>
      <p class="reason">${song.artist_name}</p>
      <div class="feature-list">
        ${Object.entries(data.feature_summary || {})
          .slice(0, 4)
          .map(([key, value]) => `<span class="pill">${key}: ${Number(value).toFixed(2)}</span>`)
          .join('')}
      </div>
      <p class="reason">${data.reason}</p>
      <p class="reason">${data.explanation}</p>
    `;

    const section = document.getElementById('search-results');
    if (section) {
      section.prepend(card);
    }
    setFeedback(searchFeedback, `Mood predicted: ${data.predicted_mood}.`, 'success');
  } catch (error) {
    setFeedback(searchFeedback, error.message, 'error');
  }
}

async function handlePlaylistUpload(event) {
  event.preventDefault();
  const fileInput = document.getElementById('playlist-file');
  const file = fileInput.files[0];

  if (!file) {
    setFeedback(playlistFeedback, 'Please choose a CSV file before analyzing.', 'error');
    return;
  }

  if (!file.name.toLowerCase().endsWith('.csv')) {
    setFeedback(playlistFeedback, 'Only CSV files are supported for playlist analysis.', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  setFeedback(playlistFeedback, 'Uploading and analyzing the playlist…');
  playlistSummary.innerHTML = '';
  playlistSummary.classList.add('hidden');
  playlistResults.innerHTML = '';

  try {
    const data = await fetchJson('/api/analyze-playlist', {
      method: 'POST',
      body: formData,
    });

    const moodDistribution = data.mood_distribution || {};
    const distributionEntries = Object.entries(moodDistribution)
      .filter(([_, value]) => Number(value) > 0)
      .map(([mood, value]) => {
        const total = Number(data.eligible_rows || 1);
        const percent = ((Number(value) / total) * 100).toFixed(1);
        return `
          <div class="mood-row">
            <span>${mood}</span>
            <div class="mood-bar"><span class="mood-fill" style="width:${percent}%"></span></div>
            <strong>${value}</strong>
          </div>
        `;
      })
      .join('');

    playlistSummary.classList.remove('hidden');
    playlistSummary.innerHTML = `
      <h3>Playlist summary</h3>
      <div class="summary-stats">
        <span>Rows: ${data.rows}</span>
        <span>Eligible songs: ${data.eligible_rows}</span>
        <span>Columns: ${data.columns.length}</span>
      </div>
      <div class="mood-chart">${distributionEntries || '<p class="reason">No eligible songs had enough valid feature data.</p>'}</div>
    `;

    if (data.results && data.results.length > 0) {
      playlistResults.innerHTML = data.results.map(renderPlaylistCard).join('');
      const csvButton = document.createElement('button');
      csvButton.type = 'button';
      csvButton.className = 'primary-button';
      csvButton.style.marginTop = '16px';
      csvButton.textContent = 'Download analysis CSV';
      csvButton.addEventListener('click', () => {
        const blob = new Blob([data.csv_download], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'tunewise_playlist_analysis.csv';
        link.click();
        URL.revokeObjectURL(url);
      });
      playlistResults.appendChild(csvButton);
    } else {
      playlistResults.innerHTML = emptyState('No songs were eligible for mood analysis. Check the CSV columns and numeric feature values.');
    }

    setFeedback(playlistFeedback, `Playlist analysis complete. ${data.eligible_rows} rows were used for mood scoring.`, 'success');
  } catch (error) {
    setFeedback(playlistFeedback, error.message, 'error');
    playlistResults.innerHTML = emptyState('The playlist could not be analyzed.');
  }
}

async function loadMoodRecommendations(mood) {
  moodResults.innerHTML = emptyState(`Finding ${mood.toLowerCase()} tracks…`);
  try {
    const data = await fetchJson(`/api/recommendations?mood=${encodeURIComponent(mood)}`);
    if (!data.results || data.results.length === 0) {
      moodResults.innerHTML = emptyState(`No songs in the local dataset matched the ${mood} mood closely enough.`);
      return;
    }
    moodResults.innerHTML = data.results.map(renderSongCard).join('');
    document.querySelectorAll('[data-song]').forEach((button) => {
      button.addEventListener('click', async () => {
        const song = JSON.parse(button.getAttribute('data-song'));
        await predictSong(song);
      });
    });
  } catch (error) {
    moodResults.innerHTML = emptyState(error.message || 'Recommendations could not be loaded.');
  }
}

async function loadModelInfo() {
  try {
    const data = await fetchJson('/api/model-info');
    modelInfo.innerHTML = `
      <article class="model-card">
        <h3>Actual method in use</h3>
        <p>${data.method}</p>
      </article>
      <article class="model-card">
        <h3>What the system assumes</h3>
        <ul>
          ${data.assumptions.map((point) => `<li>${point}</li>`).join('')}
        </ul>
      </article>
      <article class="model-card">
        <h3>Supported mood labels</h3>
        <div class="inline-pills">${data.moods.map((item) => `<span class="pill">${item}</span>`).join('')}</div>
      </article>
    `;
  } catch (error) {
    modelInfo.innerHTML = `<article class="model-card"><h3>Model info unavailable</h3><p>${error.message}</p></article>`;
  }
}

searchForm.addEventListener('submit', handleSongSearch);
playlistForm.addEventListener('submit', handlePlaylistUpload);
document.querySelectorAll('.mood-card').forEach((button) => {
  button.addEventListener('click', () => {
    const mood = button.dataset.mood;
    document.querySelectorAll('.mood-card').forEach((card) => {
      const selected = card === button;
      card.classList.toggle('is-selected', selected);
      card.setAttribute('aria-pressed', String(selected));
    });
    loadMoodRecommendations(mood);
  });
});

loadDatasetCount();
loadModelInfo();
