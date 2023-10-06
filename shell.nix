{ pkgs ? import <nixpkgs> {} }:
let
	my-packages = ps: with ps; [
		numpy
<<<<<<< HEAD
		matplotlib
=======
>>>>>>> 87f4459 (Update random_forest.py)
		pandas
	];
in pkgs.mkShell {
	packages = with pkgs; [
		(python3.withPackages my-packages)
	];
}
