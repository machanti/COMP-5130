{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs = inputs @ {
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = ["x86_64-linux"];
      perSystem = {
        pkgs,
        ...
      }: {
        devShells.default = pkgs.mkShellNoCC {
          packages = [pkgs.python.withPackages (pp: with pp; [
              numpy
              matplotlib
              pmdarima
              prophet
            ])];
        };
      };
    };
}
