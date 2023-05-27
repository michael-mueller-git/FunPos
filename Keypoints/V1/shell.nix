{ pkgs ? import <nixpkgs> {} }:
let python =
    let
    packageOverrides = self:
    super: {
      opencv4 = super.opencv4.overrideAttrs (old: rec {
        buildInputs = old.buildInputs ++ [pkgs.qt5.full];
        cmakeFlags = old.cmakeFlags ++ ["-DWITH_QT=ON"];
      });

      qudida = pkgs.python39Packages.buildPythonPackage rec {
        pname = "qudida";
        version = "0.0.4";
        format = "wheel";

        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/f0/a1/a5f4bebaa31d109003909809d88aeb0d4b201463a9ea29308d9e4f9e7655/qudida-0.0.4-py3-none-any.whl";
          sha256 = "4519714c40cd0f2e6c51e1735edae8f8b19f4efe1f33be13e9d644ca5f736dd6";
        };

        propagatedBuildInputs = with pkgs.python39Packages; [
          python.pkgs.opencv4
          scikit-learn
          typing-extensions
        ];

        pipInstallFlags = [ "--no-dependencies" ];
        doCheck = false;
      };

      albumentations = pkgs.python39Packages.buildPythonPackage rec {
        pname = "albumentations";
        version = "1.3.0";
        format = "wheel";

        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/4f/55/3c2ce84c108fc1d422afd6de153e4b0a3e6f96ecec4cb9afcf0284ce3538/albumentations-1.3.0-py3-none-any.whl";
          sha256 = "294165d87d03bc8323e484927f0a5c1a3c64b0e7b9c32a979582a6c93c363bdf";
        };

        propagatedBuildInputs = with pkgs.python39Packages; [
          scipy
          scikit-learn
          scikitimage
          python.pkgs.qudida
        ];

        pipInstallFlags = [ "--no-dependencies" ];
        doCheck = false;
      };

      simplification = pkgs.python39Packages.buildPythonPackage rec {
        pname = "simplification";
        version = "0.6.2";
        format = "wheel";

        src = pkgs.fetchurl {
          url = "https://files.pythonhosted.org/packages/02/3e/829b59a5d072feb45e14879d3149a2dad743a18f83db29d8f3800a33aa64/simplification-0.6.2-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
          sha256 = "20fb00f219bdd33319fc7526d23ef1fb7e52a40027a010f1013dd60626282325";
        };

        doCheck = false;
      };
    };
    in
      pkgs.python39.override {inherit packageOverrides; self = python;};
in
  pkgs.mkShell {
    nativeBuildInputs = with pkgs; [
      zsh
      qt5.qtbase
      qt5.full
      qt5.wrapQtAppsHook
      libsForQt5.breeze-qt5
      libsForQt5.qt5ct
      python.pkgs.opencv4
      python.pkgs.simplification
      python.pkgs.albumentations
      (python39.withPackages (p: with p; [
        coloredlogs
        cryptography
        matplotlib
        pillow
        pip
        pynput
        pyqt5
        pyqtgraph
        pycocotools
        pyyaml
        pytorch
        torchvision
        scipy
        screeninfo
      ]))
    ];

 shellHook = ''
    export QT_QPA_PLATFORM="xcb"
  '';
}
