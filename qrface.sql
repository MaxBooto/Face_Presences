-- phpMyAdmin SQL Dump
-- version 5.1.1
-- https://www.phpmyadmin.net/
--
-- Hôte : 127.0.0.1
-- Généré le : dim. 25 mai 2025 à 12:07
-- Version du serveur : 10.4.22-MariaDB
-- Version de PHP : 8.1.1

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Base de données : `qrface`
--

-- --------------------------------------------------------

--
-- Structure de la table `admin`
--

CREATE TABLE `admin` (
  `id` int(11) NOT NULL,
  `nom` varchar(255) NOT NULL,
  `prenom` varchar(255) NOT NULL,
  `mail` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Déchargement des données de la table `admin`
--

INSERT INTO `admin` (`id`, `nom`, `prenom`, `mail`, `password`) VALUES
(1, 'max', 'max', 'max@gmail.com', 'b0daae94d6f54c8966fe5b05c61dc095');

-- --------------------------------------------------------

--
-- Structure de la table `image`
--

CREATE TABLE `image` (
  `id` int(11) NOT NULL,
  `photo` varchar(255) NOT NULL,
  `id_users` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Déchargement des données de la table `image`
--

INSERT INTO `image` (`id`, `photo`, `id_users`) VALUES
(1, 'dataset/mbm\\mbm_1.jpg', 1),
(2, 'dataset/mbm\\mbm_10.jpg', 1),
(3, 'dataset/mbm\\mbm_11.jpg', 1),
(4, 'dataset/mbm\\mbm_12.jpg', 1),
(5, 'dataset/mbm\\mbm_13.jpg', 1),
(6, 'dataset/mbm\\mbm_14.jpg', 1),
(7, 'dataset/mbm\\mbm_15.jpg', 1),
(8, 'dataset/mbm\\mbm_16.jpg', 1),
(9, 'dataset/mbm\\mbm_17.jpg', 1),
(10, 'dataset/mbm\\mbm_18.jpg', 1),
(11, 'dataset/mbm\\mbm_19.jpg', 1),
(12, 'dataset/mbm\\mbm_2.jpg', 1),
(13, 'dataset/mbm\\mbm_20.jpg', 1),
(14, 'dataset/mbm\\mbm_21.jpg', 1),
(15, 'dataset/mbm\\mbm_22.jpg', 1),
(16, 'dataset/mbm\\mbm_23.jpg', 1),
(17, 'dataset/mbm\\mbm_24.jpg', 1),
(18, 'dataset/mbm\\mbm_25.jpg', 1),
(19, 'dataset/mbm\\mbm_26.jpg', 1),
(20, 'dataset/mbm\\mbm_27.jpg', 1),
(21, 'dataset/mbm\\mbm_28.jpg', 1),
(22, 'dataset/mbm\\mbm_29.jpg', 1),
(23, 'dataset/mbm\\mbm_3.jpg', 1),
(24, 'dataset/mbm\\mbm_30.jpg', 1),
(25, 'dataset/mbm\\mbm_31.jpg', 1),
(26, 'dataset/mbm\\mbm_32.jpg', 1),
(27, 'dataset/mbm\\mbm_33.jpg', 1),
(28, 'dataset/mbm\\mbm_34.jpg', 1),
(29, 'dataset/mbm\\mbm_35.jpg', 1),
(30, 'dataset/mbm\\mbm_36.jpg', 1),
(31, 'dataset/mbm\\mbm_37.jpg', 1),
(32, 'dataset/mbm\\mbm_38.jpg', 1),
(33, 'dataset/mbm\\mbm_39.jpg', 1),
(34, 'dataset/mbm\\mbm_4.jpg', 1),
(35, 'dataset/mbm\\mbm_40.jpg', 1),
(36, 'dataset/mbm\\mbm_41.jpg', 1),
(37, 'dataset/mbm\\mbm_42.jpg', 1),
(38, 'dataset/mbm\\mbm_43.jpg', 1),
(39, 'dataset/mbm\\mbm_44.jpg', 1),
(40, 'dataset/mbm\\mbm_45.jpg', 1),
(41, 'dataset/mbm\\mbm_46.jpg', 1),
(42, 'dataset/mbm\\mbm_47.jpg', 1),
(43, 'dataset/mbm\\mbm_48.jpg', 1),
(44, 'dataset/mbm\\mbm_49.jpg', 1),
(45, 'dataset/mbm\\mbm_5.jpg', 1),
(46, 'dataset/mbm\\mbm_50.jpg', 1),
(47, 'dataset/mbm\\mbm_51.jpg', 1),
(48, 'dataset/mbm\\mbm_6.jpg', 1),
(49, 'dataset/mbm\\mbm_7.jpg', 1),
(50, 'dataset/mbm\\mbm_8.jpg', 1),
(51, 'dataset/mbm\\mbm_9.jpg', 1);

-- --------------------------------------------------------

--
-- Structure de la table `presences`
--

CREATE TABLE `presences` (
  `id` int(11) NOT NULL,
  `date` varchar(10) NOT NULL,
  `heure` varchar(8) NOT NULL,
  `id_users` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Déchargement des données de la table `presences`
--

INSERT INTO `presences` (`id`, `date`, `heure`, `id_users`) VALUES
(3, '25/05/2025', '09:32:40', 1);

-- --------------------------------------------------------

--
-- Structure de la table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `nom` varchar(255) NOT NULL,
  `prenom` varchar(255) NOT NULL,
  `sexe` varchar(10) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Déchargement des données de la table `users`
--

INSERT INTO `users` (`id`, `nom`, `prenom`, `sexe`) VALUES
(1, 'mbomba', 'max', 'M');

--
-- Index pour les tables déchargées
--

--
-- Index pour la table `admin`
--
ALTER TABLE `admin`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `mail` (`mail`);

--
-- Index pour la table `image`
--
ALTER TABLE `image`
  ADD PRIMARY KEY (`id`),
  ADD KEY `id_users` (`id_users`);

--
-- Index pour la table `presences`
--
ALTER TABLE `presences`
  ADD PRIMARY KEY (`id`),
  ADD KEY `id_users` (`id_users`);

--
-- Index pour la table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT pour les tables déchargées
--

--
-- AUTO_INCREMENT pour la table `admin`
--
ALTER TABLE `admin`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT pour la table `image`
--
ALTER TABLE `image`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=152;

--
-- AUTO_INCREMENT pour la table `presences`
--
ALTER TABLE `presences`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=5;

--
-- AUTO_INCREMENT pour la table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- Contraintes pour les tables déchargées
--

--
-- Contraintes pour la table `image`
--
ALTER TABLE `image`
  ADD CONSTRAINT `image_ibfk_1` FOREIGN KEY (`id_users`) REFERENCES `users` (`id`);

--
-- Contraintes pour la table `presences`
--
ALTER TABLE `presences`
  ADD CONSTRAINT `presences_ibfk_1` FOREIGN KEY (`id_users`) REFERENCES `users` (`id`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
