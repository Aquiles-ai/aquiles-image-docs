const searchInput = document.getElementById('search-input');
const mobileSearchInput = document.getElementById('mobile-search-input');
const searchResults = document.getElementById('search-results');

// Shared Index
const searchIndex = [
    { title: "Introduction", hash: "#introduction", category: "Getting Started", snippet: "Overview of Aquiles-Image features and capabilities." },
    { title: "Installation", hash: "#installation", category: "Getting Started", snippet: "Pip install, requirements, GPU setup." },
    { title: "Prerequisites", hash: "#installation", category: "Getting Started", snippet: "Python 3.8+, CUDA GPU, VRAM requirements." },
    { title: "Launch Server", hash: "#server", category: "Guides", snippet: "CLI commands, hosting, port configuration." },
    { title: "Dev Mode", hash: "#dev-mode", category: "Guides", snippet: "Run without GPU, mock endpoints for testing." },
    { title: "Modal Deployment", hash: "#modal-deployment", category: "Guides", snippet: "Deploy serverless GPU inference on Modal." },
    { title: "Python Client", hash: "#client-api", category: "API", snippet: "OpenAI compatible client usage example." },
    { title: "Supported Models", hash: "#models", category: "API", snippet: "Stable Diffusion 3.5, FLUX, VRAM usage." },
];

function handleSearch(query) {
    if (!query || query.length < 2) {
        searchResults.classList.add('hidden');
        return;
    }

    const q = query.toLowerCase();
    const results = searchIndex.filter(item => 
        item.title.toLowerCase().includes(q) || 
        item.snippet.toLowerCase().includes(q)
    );

    if (results.length > 0) {
        searchResults.innerHTML = results.map(item => `
            <a href="${item.hash}" onclick="closeSearch()" class="block px-4 py-3 hover:bg-brand-bg transition-colors group">
                <div class="flex items-center justify-between">
                    <span class="text-sm font-medium text-gray-200 group-hover:text-brand-primary">${item.title}</span>
                    <span class="text-xs text-gray-600 border border-brand-border rounded px-1.5 py-0.5">${item.category}</span>
                </div>
                <p class="text-xs text-gray-500 mt-1 truncate">${item.snippet}</p>
            </a>
        `).join('');
    } else {
        searchResults.innerHTML = `
            <div class="px-4 py-8 text-center text-gray-500 text-sm">
                No results found for "<span class="text-gray-300">${query}</span>"
            </div>
        `;
    }
    searchResults.classList.remove('hidden');
}

function closeSearch() {
    searchResults.classList.add('hidden');
    if(searchInput) searchInput.value = '';
    if(mobileSearchInput) mobileSearchInput.value = '';
}

// Event Listeners
if (searchInput) {
    searchInput.addEventListener('input', (e) => handleSearch(e.target.value));
    searchInput.addEventListener('focus', (e) => handleSearch(e.target.value));
}

if (mobileSearchInput) {
    mobileSearchInput.addEventListener('input', (e) => handleSearch(e.target.value));
}

// Keyboard Shortcut (Ctrl/Cmd + K)
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        searchInput?.focus();
    }
    if (e.key === 'Escape') {
        closeSearch();
        searchInput?.blur();
    }
});

// Click outside to close
document.addEventListener('click', (e) => {
    if (!e.target.closest('#search-input') && !e.target.closest('#search-results') && !e.target.closest('#mobile-search-input')) {
        closeSearch();
    }
});