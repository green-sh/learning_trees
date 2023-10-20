{ pkgs ? import <nixpkgs> {} }:
let
	my-packages = ps: with ps; [
		numpy
		matplotlib
		pandas
	];
in pkgs.mkShell {
	packages = with pkgs; [
		(python3.withPackages my-packages)
	];
}
