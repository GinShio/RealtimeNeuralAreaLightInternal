{
  description = "Realtime Neural Area Light";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    fenix.url = "github:nix-community/fenix";
    crane.url = "github:ipetkov/crane";
  };

  outputs = { self, nixpkgs, flake-utils, fenix, crane }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ fenix.overlays.default ];
        };

        craneLib =
          (crane.mkLib pkgs).overrideToolchain (p: p.fenix.stable.toolchain);

        slangc = with pkgs; [ shader-slang ];

        vulkanPackages = with pkgs; [ vulkan-loader vulkan-validation-layers ];

        x11Packages = with pkgs; [
          libxkbcommon
          mesa
          pkg-config
          xorg.libX11
          xorg.libXcursor
          xorg.libXi
          xorg.libXrandr
          xorg.libXext
          xorg.libxcb
          xorg.libXrender
          xorg.libXfixes
          xorg.xcbutil
          xorg.xcbutilwm
          xorg.xcbutilimage
          xorg.xcbutilkeysyms
          xorg.xcbutilrenderutil
          xkeyboard_config
        ];

        crates = pkgs.callPackage ./nix/crates.nix {
          inherit craneLib slangc x11Packages vulkanPackages;
        };

        shell = pkgs.callPackage ./nix/shell.nix {
          inherit craneLib slangc x11Packages vulkanPackages;
        };
      in {
        devShells.default = shell.devShell;
        packages = crates.packages;
        checks = crates.checks;
      });
}
