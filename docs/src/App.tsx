import { useState, useEffect, useMemo } from 'react';
import Fuse from 'fuse.js';
import { Search, X } from 'lucide-react';
import AppSwitcher from './AppSwitcher';
import defaultDocs from "../search.json";
import Features from './Features';
import QrispLogo from './img/qrisp_logo.png'
import { versionConfigs, versionConfigsPromise, type VersionType, type VersionConfig } from './configs';

interface Doc {
  title: string;
  package: string;
  description: string;
  url: string;
}

function App() {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [docs, setDocs] = useState<Doc[]>(defaultDocs);
  const [searchResults, setSearchResults] = useState<Doc[]>(defaultDocs);
  const [selectedVersion, setSelectedVersion] = useState<VersionType>('');
  const [availableVersions, setAvailableVersions] = useState<VersionConfig[]>(versionConfigs);
  const [isLoadingVersions, setIsLoadingVersions] = useState(true);

  // Load dynamic version configurations
  useEffect(() => {
    const loadVersions = async () => {
      try {
        const dynamicConfigs = await versionConfigsPromise;
        if (dynamicConfigs.length > 0) {
          setAvailableVersions(dynamicConfigs);

          // Set default version to the one marked as default, or the first one
          const defaultConfig = dynamicConfigs.find(v => v.isDefault) || dynamicConfigs[0];
          setSelectedVersion(defaultConfig.id);
        }
      } catch (error) {
        console.warn('Failed to load dynamic version configs, using static fallback:', error);
      } finally {
        setIsLoadingVersions(false);
      }
    };

    loadVersions();
  }, []);

  const currentVersionConfig = availableVersions.find(v => v.id === selectedVersion) || availableVersions[0];

  // Load search index for the selected version
  useEffect(() => {
    const loadSearchIndex = async () => {
      try {
        let searchIndexUrl = './search.json'; // Default for resonance

        if (selectedVersion !== 'resonance') {
          // Map version ID to directory name
          const pathPrefix = currentVersionConfig.pathPrefix;
          const dirName = pathPrefix.replace('./', '').replace('/', ''); // Convert './sdk4_1/' to 'sdk4_1'
          searchIndexUrl = `./${dirName}/search_${dirName}.json`;
        }

        const response = await fetch(searchIndexUrl);
        if (response.ok) {
          const versionDocs = await response.json();
          setDocs(versionDocs);
          setSearchResults(versionDocs);
        } else {
          console.warn(`Failed to load search index for ${selectedVersion}, falling back to default`);
          setDocs(defaultDocs);
          setSearchResults(defaultDocs);
        }
      } catch (error) {
        console.warn(`Error loading search index for ${selectedVersion}:`, error);
        setDocs(defaultDocs);
        setSearchResults(defaultDocs);
      }
    };

    loadSearchIndex();
  }, [selectedVersion, currentVersionConfig]);

  // Get packages for a specific version
  const getPackagesForVersion = (versionId: VersionType): string[] => {
    const config = availableVersions.find(v => v.id === versionId);
    return config ? config.packages : [];
  };

  // Initialize version from URL on component mount
  useEffect(() => {
    if (isLoadingVersions || availableVersions.length === 0) return;

    const urlParams = new URLSearchParams(window.location.search);
    const versionFromUrl = urlParams.get('version') as VersionType;
    if (versionFromUrl && availableVersions.some(v => v.id === versionFromUrl)) {
      setSelectedVersion(versionFromUrl);
    }
  }, [isLoadingVersions, availableVersions]);

  // Update URL when version changes
  useEffect(() => {
    if (!selectedVersion || isLoadingVersions) return;

    const url = new URL(window.location.href);
    const defaultConfig = availableVersions.find(v => v.isDefault);

    if (selectedVersion !== defaultConfig?.id) {
      url.searchParams.set('version', selectedVersion);
    } else {
      url.searchParams.delete('version');
    }
    window.history.replaceState({}, '', url.toString());
  }, [selectedVersion, availableVersions, isLoadingVersions]);

  const fuse = useMemo(() => new Fuse(docs, {
    keys: ['title', 'description', 'package'],
    threshold: 0.4
  }), [docs]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check for CMD+K (Mac) or CTRL+K (Windows/Linux)
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsModalOpen(true);
      }
      // Close modal on ESC
      if (e.key === 'Escape') {
        setIsModalOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    if (searchQuery) {
      const results = fuse.search(searchQuery);
      setSearchResults(results.map(result => result.item));
    } else {
      setSearchResults(docs);
    }
  }, [searchQuery, fuse, docs]);

  const handleSearchClick = () => {
    setIsModalOpen(true);
  };

  const [isDocumentationSelected, setIsDocumentationSelected] = useState(true);

  // Base documentation entries with their descriptions
  const baseDocLinks = [
    { package: "iqm-pulla", title: "IQM Pulla", description: "Pulse-level access library for compiling quantum circuits." },
    { package: "iqm-benchmarks", title: "IQM Benchmarks", description: "Quantum Characterization, Verification, and Validation (QCVV) tools for quantum computing." },
    { package: "iqm-pulse", title: "IQM Pulse", description: "Interface and implementations for control pulses." },
    { package: "iqm-qaoa", title: "IQM QAOA", description: "Easily set up and run different flavours of QAOA." },
    { package: "iqm-client", title: "IQM Client", description: "Python client for remote access to quantum computers for circuit-level access (e.g. via Qiskit, Cirq)." },
    { package: "iqm-station-control-client", title: "IQM Station Control Client", description: "Python client for remote access to quantum computers for pulse-level access." },
    { package: "iqm-exa-common", title: "IQM EXA Common", description: "Abstract interfaces, helpers, utility classes, etc." },
    { package: "iqm-data-definitions", title: "IQM Data Definitions", description: "A common place for data definitions shared inside IQM." },
    { package: "qrisp", title: "Qrisp", description: "Use Eclipse Qrisp to run your circuits on IQM hardware.", external: "https://qrisp.eu/reference/index.html", image: QrispLogo },
    { package: "qiskit-iqm", title: "Qiskit on IQM", description: "Qiskit provider for accessing IQM quantum computers. Only used up to IQM OS 3.4"},
    { package: "cirq-iqm", title: "Cirq on IQM", description: "Cirq provider for accessing IQM quantum computers. Only used up to IQM OS 3.4"},
    { package: "iqm-qubit-selector", title: "IQM Qubit Selector", description: "Tools for selecting optimal qubits for quantum circuits." }

  ];

  // Generate docLinks based on selected version and available packages
  const getDocLinks = () => {
    const availablePackages = getPackagesForVersion(selectedVersion);
    const pathPrefix = currentVersionConfig.pathPrefix;

    return baseDocLinks
      .filter(doc => {
        // For internal docs, check if package is available in the current version
        return availablePackages.length === 0 || availablePackages.includes(doc.package);
      })
      .map(doc => ({
        ...doc,
        href: doc.external || `${pathPrefix}${doc.package}${doc.package === 'iqm-client' ? '/' : ''}`
      }));
  };

  const docLinks = getDocLinks();

  return (
    <div className="min-h-screen px-8 py-3">
      <div className="mx-auto">

        <div className="flex flex-col sm:flex-row mb-4 sm:gap-2 lg:gap-[8rem]">
          <AppSwitcher />

          <div className="flex gap-4">
            <button
              className="relative px-4 pt-2"
              onClick={() => setIsDocumentationSelected(true)}
            >
              Documentation
              <span className={`block h-[0.2rem] ml-4 mr-4 ${isDocumentationSelected ? 'bg-[#69ded7]' : 'bg-transparent'} absolute bottom-0 left-0 right-0`}></span>
            </button>
            <button
              className="relative px-4 pt-2"
              onClick={() => setIsDocumentationSelected(false)}
            >
              Features
                <span className={`block h-[0.2rem] ml-4 mr-4 ${!isDocumentationSelected ? 'bg-[#69ded7]' : 'bg-transparent'} absolute bottom-0 left-0 right-0`}></span>
            </button>
          </div>

        </div>

        <div className='max-w-4xl mx-auto'>
          <div
            onClick={handleSearchClick}
            className="mt-6 mb-6 flex items-center gap-2 p-3 bg-white border border-gray-200 rounded-lg cursor-pointer hover:border-gray-300 transition-colors"
          >
            <Search className="w-5 h-5 text-gray-400" />
            <span className="text-gray-500">Search all documentation... {navigator.userAgent.includes('Mac') ? "Press ⌘K" : "Press Ctrl+K"}</span>
          </div>
          {isDocumentationSelected ? (
            <>
              <p>Find below the documentation for IQM client-side libraries that can be used to connect to {" "}
                <a href="https://resonance.meetiqm.com" target="_blank">IQM Resonance</a> and any IQM on-premise quantum computer.
                {currentVersionConfig && currentVersionConfig.description && (
                  <span className={`block mt-2 text-sm p-2 rounded ${
                    currentVersionConfig.isPreview
                      ? 'text-orange-600 bg-orange-50 border border-orange-200'
                      : 'text-blue-600 bg-blue-50 border border-blue-200'
                  }`}>
                    {currentVersionConfig.description}
                  </span>
                )}
              </p>
              {isLoadingVersions ? (
                <div className="flex justify-center items-center mt-8 py-12">
                  <div className="text-gray-500">Loading version configurations...</div>
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mt-8">
                  {docLinks.map((doc, index) => (
                      <a key={index} href={doc.href} target='_blank' className="p-6 bg-white border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition-shadow relative">
                      {doc.image ? (
                        <img
                        src={doc.image}
                        className="h-7"
                        alt={doc.title + " logo"}
                        />
                      ) :
                      <h2 className="text-lg font-semibold text-gray-900 pr-10">{doc.title}</h2>}
                      <p className="mt-2 text-sm text-gray-600">{doc.description}</p>
                      </a>
                  ))}
                </div>
              )}
            </>
          ) : (
            <Features />
          )}

          {/* Modal */}
          {isModalOpen && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-start justify-center pt-[15vh] z-50">
              <div className="bg-white rounded-xl shadow-2xl w-full max-w-2xl max-h-[70vh] overflow-hidden">
                <div className="p-4 border-b border-gray-100 flex items-center gap-3">
                  <Search className="w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    autoFocus
                    placeholder="Search documentation..."
                    className="flex-1 outline-none text-gray-900"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                  />
                  <button
                    onClick={() => setIsModalOpen(false)}
                    className="p-1 hover:bg-gray-100 rounded-md transition-colors"
                  >
                    <X className="w-5 h-5 text-gray-500" />
                  </button>
                </div>

                <div className="overflow-y-auto max-h-[calc(70vh-4rem)]">
                  {searchResults.map((doc, index) => (
                    <a
                      key={index}
                      href={"." + doc.url}
                      target="_blank"
                      className="block p-4 hover:bg-gray-50 transition-colors overflow-hidden"
                    >
                      <h3 className="font-medium text-gray-900">{doc.title}</h3>
                      <span className="text-sm text-gray-500 block mt-1">
                        {doc.package}
                      </span>
                      <p className="text-sm text-gray-600 mt-1">
                        {doc.description}
                      </p>
                    </a>
                  ))}
                </div>
              </div>
            </div>
          )}

        </div>

        {/* Floating Version Selector */}
        {!isLoadingVersions && availableVersions.length > 0 && (
          <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 z-40">
            <div className="bg-white/90 backdrop-blur-md border border-gray-200 rounded-2xl shadow-lg px-2 py-2">
              <div className="flex gap-1">
                {availableVersions.map((version) => (
                  <button
                    key={version.id}
                    className={`px-3 py-2 text-xs sm:text-sm rounded-xl transition-all duration-200 ${
                    selectedVersion === version.id
                      ? 'text-white shadow-md transform scale-105'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-800'
                    }`}
                    style={selectedVersion === version.id ? {
                    background: version.isPreview
                      ? 'linear-gradient(45deg, #9ca3af, #6b7280)'
                      : 'linear-gradient(45deg, #759deb, #5fdd97)'
                    } : {}}
                    onClick={() => setSelectedVersion(version.id)}
                    title={version.isPreview ? 'Preview version' : version.isDefault ? 'Default version' : ''}
                  >
                    <span className="items-center gap-1">
                      {version.label}
                      {version.isPreview && <sup className="text-xxs text-gray-400">PREVIEW</sup>}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}

        <footer className="mt-8 text-center text-sm text-gray-500 border-gray-300 border-t pt-4 pb-20">
          <span>Copyright IQM Quantum Computers 2021-2026.</span>
          <br />
          <span>Need assistance? Contact us <a href="mailto:support@meetiqm.com">support@meetiqm.com</a></span>
        </footer>
      </div>
    </div>
  );
}

export default App;
