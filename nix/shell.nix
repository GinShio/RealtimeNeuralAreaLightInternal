{ pkgs, craneLib, slangc, x11Packages, vulkanPackages }:
let
  # vulkan env vars
  vulkanLayerPath =
    "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
  nvidiaIcdFile =
    "/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.x86_64.json";
  libraryPath = pkgs.lib.makeLibraryPath
    (with pkgs; [ libxkbcommon mesa vulkan-loader stdenv.cc.cc ]);
in {
  devShell = craneLib.devShell {
    buildInputs = with pkgs;
      [ mold clang stdenv ] ++ slangc ++ vulkanPackages ++ x11Packages;

    # use x11 instead of wayland
    shellHook = "unset WAYLAND_DISPLAY";

    # set the Vulkan environment variables
    env = {
      RUST_BACKTRACE = 1;
      VK_ICD_FILENAMES = nvidiaIcdFile;
      VK_LAYER_PATH = vulkanLayerPath;
      LD_LIBRARY_PATH = "${libraryPath}:$LD_LIBRARY_PATH";
    };
  };
}
