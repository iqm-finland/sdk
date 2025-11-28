export type VersionType = string;

export interface VersionConfig {
  id: VersionType;
  label: string;
  pathPrefix: string;
  description?: string;
  packages: string[];
  isDefault?: boolean;
  isPreview?: boolean;
}

// Function to extract packages from SDK file content
function parsePackagesFromSdkContent(content: string): string[] {
  return content
    .split('\n')
    .map(line => line.trim())
    .filter(line => line && !line.startsWith('#'))
    .map(line => {
      // Extract package name before any brackets or version specifiers
      const match = line.match(/^([a-zA-Z0-9-_]+)/);
      return match ? match[1] : line;
    })
    .filter(pkg => pkg);
}

// Function to discover SDK files dynamically
async function discoverSdkFiles(): Promise<{filename: string, path: string}[]> {
  const sdkFiles: {filename: string, path: string}[] = [];
  
  // Get max versions from environment variables or use defaults
  const maxMajor = parseInt(import.meta.env.VITE_SDK_MAX_MAJOR_VERSION || '10', 10);
  const maxMinor = parseInt(import.meta.env.VITE_SDK_MAX_MINOR_VERSION || '10', 10);
  
  console.log(`🔍 Scanning for SDK files up to version ${maxMajor}.${maxMinor}`);
  
  // Try to fetch from different locations, prioritizing public folder
  const possibleBasePaths = ['./public/', '../', './'];
  
  for (const basePath of possibleBasePaths) {
    const foundInThisPath: {filename: string, path: string}[] = [];
    
    // Scan based on environment-configured or default limits
    for (let major = 3; major <= maxMajor; major++) {
      for (let minor = 0; minor <= maxMinor; minor++) {
        const patterns = [`sdk${major}_${minor}.txt`, `sdk${major}_${minor}_default.txt`];
        
        for (const filename of patterns) {
          try {
            const fullPath = `${basePath}${filename}`;
            const response = await fetch(fullPath);
            
            // Be very strict about what constitutes a valid response
            if (response.ok && response.status === 200 && response.headers.get('content-type')?.includes('text')) {
              const text = await response.text();
              
              // Very strict validation for SDK files
              if (text.trim().length > 10 && // Must have substantial content
                  !text.includes('404') && 
                  !text.includes('Not Found') &&
                  !text.includes('<html>') &&
                  !text.includes('<HTML>') &&
                  !text.includes('<!DOCTYPE') &&
                  text.split('\n').some(line => line.trim().match(/^[a-zA-Z0-9-_]+(\[.*?\])?(\s*==.*)?$/))) { // Must contain package-like lines
                
                foundInThisPath.push({ filename, path: fullPath });
                console.log(`✓ Found valid SDK file: ${filename}`);
              } else {
                console.log(`✗ Invalid content in ${filename}: too short or doesn't look like SDK file`);
              }
            } else {
              console.log(`✗ Failed to fetch ${filename}: status ${response.status}`);
            }
          } catch {
            // File doesn't exist, this is expected for most combinations
            console.log(`✗ ${filename} not found at ${basePath}`);
          }
        }
      }
    }
    
    // If we found files in this path, use them and stop searching other paths
    if (foundInThisPath.length > 0) {
      sdkFiles.push(...foundInThisPath);
      console.log(`✓ Found ${foundInThisPath.length} valid SDK files in ${basePath}:`, foundInThisPath.map(f => f.filename));
      break;
    } else {
      console.log(`✗ No valid SDK files found in ${basePath}`);
    }
  }
  
  console.log(`📁 Total SDK files discovered: ${sdkFiles.length}`, sdkFiles.map(f => f.filename));
  return sdkFiles;
}

// Function to generate version configurations from SDK files
async function generateVersionConfigs(): Promise<VersionConfig[]> {
  const configs: VersionConfig[] = [];
  
  // Dynamically discover SDK files
  const sdkFiles = await discoverSdkFiles();
  
  if (sdkFiles.length === 0) {
    console.warn('No SDK files found, falling back to static configuration');
    return [];
  }

  let defaultVersion: string | null = null;

  for (const { filename, path } of sdkFiles) {
    try {
      const response = await fetch(path);
      if (!response.ok) continue;
      
      const content = await response.text();
      
      const packages = parsePackagesFromSdkContent(content);
      
      // Extract version info from filename
      const versionMatch = filename.match(/sdk(\d+)_(\d+)(_default)?\.txt/);
      if (!versionMatch) continue;
      
      const [, major, minor, isDefaultSuffix] = versionMatch;
      const versionKey = `${major}_${minor}`;
      const isDefault = !!isDefaultSuffix;
      
      if (isDefault) {
        defaultVersion = versionKey;
      }
      
      const config: VersionConfig = {
        id: versionKey,
        label: `IQM OS ${major}.${minor}${isDefault ? ' (Resonance)' : ''}`,
        pathPrefix: isDefault ? './' : `./sdk${major}_${minor}/`,
        packages,
        isDefault,
        isPreview: false
      };
      
      configs.push(config);
    } catch (error) {
      console.warn(`Failed to load ${filename}:`, error);
    }
  }
  
  // Sort by version (newest first)
  configs.sort((a, b) => {
    const [aMajor, aMinor] = a.id.split('_').map(Number);
    const [bMajor, bMinor] = b.id.split('_').map(Number);
    
    if (aMajor !== bMajor) return bMajor - aMajor;
    return bMinor - aMinor;
  });
  
  // Mark versions newer than default as preview
  if (defaultVersion) {
    const [defaultMajor, defaultMinor] = defaultVersion.split('_').map(Number);
    configs.forEach(config => {
      if (!config.isDefault) {
        const [major, minor] = config.id.split('_').map(Number);
        const isNewer = major > defaultMajor || (major === defaultMajor && minor > defaultMinor);
        
        if (isNewer) {
          config.isPreview = true;
          config.description = `⚠️ You are viewing preview documentation for IQM OS ${major}.${minor}. This version may contain experimental features and is subject to change.`;
        } else {
          config.description = `You are viewing documentation for IQM OS ${major}.${minor}. This version applies for on-premises installations.`;
        }
      }
    });
  }
  
  return configs;
}

// Export a promise that resolves to the version configs
export const versionConfigsPromise = generateVersionConfigs();

// For backwards compatibility, export the old static configs initially
// These will be replaced by the dynamic ones once loaded
export const versionConfigs: VersionConfig[] = [
  {
    id: '4_3',
    label: 'IQM OS 4.3 (Resonance)',
    pathPrefix: './',
    isDefault: true,
    isPreview: false,
    packages: [
      'iqm-data-definitions',
      'iqm-exa-common',
      'iqm-station-control-client',
      'iqm-pulse',
      'iqm-pulla',
      'iqm-client',
      'iqm-qaoa',
      'qrisp',
      'iqm-benchmarks'
    ]
  }
];