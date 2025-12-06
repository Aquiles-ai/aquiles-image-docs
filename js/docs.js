
const routes = {
    '': 'docs/introduction.md',
    '#introduction': 'docs/introduction.md',
    '#installation': 'docs/installation.md',
    '#server': 'docs/server.md',
    '#dev-mode': 'docs/dev-mode.md',
    '#client-api': 'docs/client-api.md',
    '#models': 'docs/models.md'
};


const routeOrder = [
    '#introduction', '#installation', '#server', '#dev-mode', '#client-api', '#models'
];
const routeTitles = {
    '#introduction': 'Introduction',
    '#installation': 'Installation',
    '#server': 'Launch Server',
    '#dev-mode': 'Dev Mode',
    '#client-api': 'Python Client',
    '#models': 'Supported Models'
};

// Sidebar Structure
const navStructure = [
    { title: 'Getting Started', items: [
        { label: 'Introduction', hash: '#introduction' },
        { label: 'Installation', hash: '#installation' }
    ]},
    { title: 'Guides', items: [
        { label: 'Launch Server', hash: '#server' },
        { label: 'Dev Mode', hash: '#dev-mode' }
    ]},
    { title: 'API Reference', items: [
        { label: 'Python Client', hash: '#client-api' },
        { label: 'Supported Models', hash: '#models' }
    ]}
];

const contentDiv = document.getElementById('content');
const loader = document.getElementById('loader');
const navContainer = document.getElementById('nav-container');
const tocContainer = document.getElementById('toc');
const pageNavContainer = document.getElementById('page-nav');

// Init
document.addEventListener('DOMContentLoaded', () => {
    renderSidebar();
    handleHashChange();
    window.addEventListener('hashchange', handleHashChange);
});

function renderSidebar() {
    navContainer.innerHTML = navStructure.map(section => `
        <div>
            <h4 class="px-3 mb-2 text-xs font-semibold text-gray-500 uppercase tracking-wider font-mono">${section.title}</h4>
            <div class="space-y-0.5">
                ${section.items.map(item => `
                    <a href="${item.hash}" class="nav-link block px-3 py-1.5 text-sm text-gray-400 hover:text-gray-200 hover:bg-brand-surface/50 rounded-r-md transition-colors duration-200" data-hash="${item.hash}">
                        ${item.label}
                    </a>
                `).join('')}
            </div>
        </div>
    `).join('');
}

async function handleHashChange() {
    let hash = window.location.hash;
    if (!routes[hash]) hash = '#introduction';

    // 1. Update Sidebar Active State
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active', 'text-brand-primary', 'font-medium');
        if (link.getAttribute('data-hash') === hash) {
            link.classList.add('active', 'text-brand-primary', 'font-medium');
        }
    });

    // 2. Load Content
    await loadMarkdown(routes[hash]);
    
    // 3. Update Prev/Next Buttons
    renderPageNav(hash);

    // 4. Scroll to top
    document.getElementById('main-scroll').scrollTop = 0;
    
    // 5. Mobile: Close sidebar
    if (window.innerWidth < 1024) {
        document.getElementById('sidebar').classList.add('-translate-x-full');
        document.getElementById('overlay').classList.add('hidden');
    }
}

async function loadMarkdown(filePath) {
    contentDiv.classList.add('opacity-50');
    loader.classList.remove('hidden');
    tocContainer.innerHTML = ''; // Clear TOC

    try {
        const response = await fetch(filePath);
        if (!response.ok) throw new Error('File not found');
        const text = await response.text();
        
        // Parse MD
        contentDiv.innerHTML = marked.parse(text);
        
        // Post-Processing
        hljs.highlightAll();
        generateTOC();
        styleTables();

    } catch (error) {
        contentDiv.innerHTML = `<div class="text-red-500">Error loading docs: ${error.message}</div>`;
    } finally {
        contentDiv.classList.remove('opacity-50');
        loader.classList.add('hidden');
    }
}

// Generate Table of Contents from H2 and H3
function generateTOC() {
    const headers = contentDiv.querySelectorAll('h2, h3');
    if (headers.length === 0) {
        tocContainer.innerHTML = '<span class="text-sm text-gray-600 pl-4">No sections</span>';
        return;
    }

    let html = '';
    headers.forEach((header, index) => {
        // Create ID if missing
        if (!header.id) {
            header.id = header.innerText.toLowerCase().replace(/[^a-z0-9]+/g, '-');
        }

        const isH3 = header.tagName === 'H3';
        const padding = isH3 ? 'pl-6' : 'pl-4';
        const fontSize = isH3 ? 'text-[13px]' : 'text-sm';
        
        html += `
            <a href="#${header.id}" class="toc-link block ${padding} py-1 ${fontSize} text-gray-500 hover:text-gray-300 border-l border-transparent hover:border-gray-500 transition-colors" onclick="document.getElementById('${header.id}').scrollIntoView({behavior: 'smooth'})">
                ${header.innerText}
            </a>
        `;
    });
    tocContainer.innerHTML = html;

    // Optional: Intersection Observer for TOC active state could go here
}

function renderPageNav(currentHash) {
    if (currentHash === '') currentHash = '#introduction';
    const currentIndex = routeOrder.indexOf(currentHash);
    
    const prev = currentIndex > 0 ? routeOrder[currentIndex - 1] : null;
    const next = currentIndex < routeOrder.length - 1 ? routeOrder[currentIndex + 1] : null;

    let html = '';
    
    if (prev) {
        html += `
            <a href="${prev}" class="group border border-brand-border rounded-lg p-4 hover:border-brand-primary transition-colors text-left">
                <div class="text-xs text-gray-500 mb-1">Previous</div>
                <div class="text-brand-primary font-medium group-hover:text-brand-secondary transition-colors">${routeTitles[prev]}</div>
            </a>
        `;
    } else { html += '<div></div>'; }

    if (next) {
        html += `
            <a href="${next}" class="group border border-brand-border rounded-lg p-4 hover:border-brand-primary transition-colors text-right">
                <div class="text-xs text-gray-500 mb-1">Next</div>
                <div class="text-brand-primary font-medium group-hover:text-brand-secondary transition-colors">${routeTitles[next]}</div>
            </a>
        `;
    }

    pageNavContainer.innerHTML = html;
}

function styleTables() {
    // Add styling wrapper for horizontal scroll on tables
    const tables = document.querySelectorAll('table');
    tables.forEach(table => {
        const wrapper = document.createElement('div');
        wrapper.className = 'overflow-x-auto my-6 border border-brand-border rounded-lg';
        table.parentNode.insertBefore(wrapper, table);
        wrapper.appendChild(table);
        
        table.classList.add('w-full', 'text-left', 'border-collapse');
        table.querySelectorAll('th').forEach(th => th.className = 'p-3 border-b border-brand-border bg-brand-surface text-gray-200 text-sm font-semibold');
        table.querySelectorAll('td').forEach(td => td.className = 'p-3 border-b border-brand-border text-gray-400 text-sm');
        table.querySelectorAll('tr:last-child td').forEach(td => td.classList.remove('border-b'));
    });
}