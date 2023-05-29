{ pkgs ? import <nixpkgs> {}}:
let
  fhs = pkgs.buildFHSUserEnv {
    name = "my-fhs-environment";

    targetPkgs = _: [
      pkgs.bash
      pkgs.bash-completion
      pkgs.micromamba
    ];


    profile = ''
      eval "$(micromamba shell hook -s bash)"
      export MAMBA_ROOT_PREFIX=${builtins.getEnv "PWD"}/.mamba
      micromamba create -y -q -n my-mamba-environment
      micromamba activate my-mamba-environment
    '';
  };
in fhs.env
