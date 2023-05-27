{
  description = "Keypoint Model 1 Environment";

  inputs = {
    nixpkgs = {
        url = "github:NixOS/nixpkgs/nixos-22.05";
      };
      flake-utils = {
        url = "github:numtide/flake-utils";
      };
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem
    (system:
    let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = false; # disable for now take to long to build
      };
      pkgs-no-cuda = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = false;
      };
    in
        {
          devShells.default = import ./shell.nix { inherit pkgs pkgs-no-cuda; };
        }
      );
}
