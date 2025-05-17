{ lib, pkgs, craneLib, slangc, x11Packages, vulkanPackages }:
let
  # === vulkan env wrapper ===
  # Vulkan env vars
  vulkanLayerPath =
    "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
  nvidiaIcdFile =
    "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json";
  libraryPath = pkgs.lib.makeLibraryPath
    (with pkgs; [ libxkbcommon mesa vulkan-loader stdenv.cc.cc ]);

  # Wrap the Vulkan environment variables around the binary
  wrapWithVulkanEnv = binaryPath: ''
    wrapProgram ${binaryPath} \
      --prefix LD_LIBRARY_PATH : ${libraryPath} \
      --set-default VK_ICD_FILENAMES ${nvidiaIcdFile} \
      --set VK_LAYER_PATH ${vulkanLayerPath} \
      --set RUST_BACKTRACE 1 \
      --run 'unset WAYLAND_DISPLAY;'
  '';

  # === common cargo args ===
  # Cargo src paths
  root = ../.;
  src = lib.fileset.toSource {
    inherit root;
    fileset = lib.fileset.unions [
      (craneLib.fileset.commonCargoSources root)
      (lib.fileset.fileFilter (file: file.hasExt "slang") root)
    ];
  };

  # cargo common args
  commonArgs = {
    inherit src;
    strictDeps = true;
    buildInputs = vulkanPackages ++ x11Packages;
    nativeBuildInputs = with pkgs;
      [ mold clang makeWrapper stdenv ] ++ slangc ++ x11Packages;
    preBuild = "ulimit -s unlimited";
  };

  # === renderer crate ===
  # Build the cargo deps only artifacts
  rendererDeps = craneLib.buildDepsOnly (commonArgs // { pname = "renderer"; });

  # Build the renderer binary crate
  renderer = craneLib.buildPackage (commonArgs // {
    cargoArtifacts = rendererDeps;
    pname = "renderer";
    postInstall = wrapWithVulkanEnv "$out/bin/renderer";
  });

  # Check clippy warnings
  rendererClippy = craneLib.cargoClippy (commonArgs // {
    cargoArtifacts = rendererDeps;
    pname = "renderer";
    cargoClippyExtraArgs = "--all-targets";
  });

in {
  packages = {
    inherit renderer;
    default = renderer;
  };
  checks = { inherit rendererClippy; };
}
