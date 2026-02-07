"""
Dataset module for the Book Recommendation System.
Generates synthetic book and user-rating data for demonstration.
"""

import pandas as pd
import numpy as np
import os

BOOKS_CSV = "data/books.csv"
RATINGS_CSV = "data/ratings.csv"


def generate_books():
    """Generate a synthetic books dataset with realistic attributes."""
    books = [
        # Fiction
        {"book_id": 1, "title": "To Kill a Mockingbird", "author": "Harper Lee", "genre": "Fiction", "year": 1960, "description": "A classic novel about racial injustice and childhood innocence in the American South through the eyes of young Scout Finch."},
        {"book_id": 2, "title": "1984", "author": "George Orwell", "genre": "Fiction", "year": 1949, "description": "A dystopian novel set in a totalitarian society ruled by Big Brother, exploring themes of surveillance, truth, and freedom."},
        {"book_id": 3, "title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "genre": "Fiction", "year": 1925, "description": "A tale of wealth, love, and the American Dream set in the Jazz Age, following the mysterious millionaire Jay Gatsby."},
        {"book_id": 4, "title": "Pride and Prejudice", "author": "Jane Austen", "genre": "Fiction", "year": 1813, "description": "A romantic novel about the complex relationship between Elizabeth Bennet and Mr. Darcy in Regency-era England."},
        {"book_id": 5, "title": "The Catcher in the Rye", "author": "J.D. Salinger", "genre": "Fiction", "year": 1951, "description": "A coming-of-age story following teenager Holden Caulfield as he navigates alienation and loss in New York City."},
        # Science Fiction
        {"book_id": 6, "title": "Dune", "author": "Frank Herbert", "genre": "Science Fiction", "year": 1965, "description": "An epic science fiction saga set on the desert planet Arrakis, exploring politics, religion, and ecology in a far future universe."},
        {"book_id": 7, "title": "Ender's Game", "author": "Orson Scott Card", "genre": "Science Fiction", "year": 1985, "description": "A young genius is recruited to a military academy in space to prepare for an alien invasion, facing moral dilemmas along the way."},
        {"book_id": 8, "title": "The Hitchhiker's Guide to the Galaxy", "author": "Douglas Adams", "genre": "Science Fiction", "year": 1979, "description": "A comedic science fiction adventure following Arthur Dent across the galaxy after Earth is demolished for a hyperspace bypass."},
        {"book_id": 9, "title": "Neuromancer", "author": "William Gibson", "genre": "Science Fiction", "year": 1984, "description": "A groundbreaking cyberpunk novel about a washed-up computer hacker hired for the ultimate hack in a dystopian future."},
        {"book_id": 10, "title": "Foundation", "author": "Isaac Asimov", "genre": "Science Fiction", "year": 1951, "description": "A mathematician predicts the fall of the Galactic Empire and establishes a foundation to preserve knowledge and shorten the dark age."},
        # Mystery / Thriller
        {"book_id": 11, "title": "The Girl with the Dragon Tattoo", "author": "Stieg Larsson", "genre": "Mystery", "year": 2005, "description": "A journalist and a brilliant hacker investigate a decades-old disappearance, uncovering dark family secrets and corruption."},
        {"book_id": 12, "title": "Gone Girl", "author": "Gillian Flynn", "genre": "Mystery", "year": 2012, "description": "A psychological thriller about a wife's disappearance and the dark secrets lurking beneath a seemingly perfect marriage."},
        {"book_id": 13, "title": "The Da Vinci Code", "author": "Dan Brown", "genre": "Mystery", "year": 2003, "description": "A symbologist unravels a trail of clues hidden in the works of Leonardo da Vinci, leading to a shocking historical secret."},
        {"book_id": 14, "title": "And Then There Were None", "author": "Agatha Christie", "genre": "Mystery", "year": 1939, "description": "Ten strangers are lured to an isolated island where they are killed one by one according to a nursery rhyme."},
        {"book_id": 15, "title": "The Silent Patient", "author": "Alex Michaelides", "genre": "Mystery", "year": 2019, "description": "A famous painter shoots her husband and then never speaks again; a psychotherapist becomes obsessed with uncovering her motive."},
        # Fantasy
        {"book_id": 16, "title": "Harry Potter and the Sorcerer's Stone", "author": "J.K. Rowling", "genre": "Fantasy", "year": 1997, "description": "A young orphan discovers he is a wizard and begins his education at Hogwarts School of Witchcraft and Wizardry."},
        {"book_id": 17, "title": "The Lord of the Rings", "author": "J.R.R. Tolkien", "genre": "Fantasy", "year": 1954, "description": "An epic fantasy quest to destroy the One Ring and save Middle-earth from the Dark Lord Sauron."},
        {"book_id": 18, "title": "A Game of Thrones", "author": "George R.R. Martin", "genre": "Fantasy", "year": 1996, "description": "Noble families vie for control of the Iron Throne in a sprawling epic of political intrigue, war, and supernatural threats."},
        {"book_id": 19, "title": "The Name of the Wind", "author": "Patrick Rothfuss", "genre": "Fantasy", "year": 2007, "description": "A legendary figure tells the true story of his life, from orphaned troupe performer to the most notorious wizard of his age."},
        {"book_id": 20, "title": "The Hobbit", "author": "J.R.R. Tolkien", "genre": "Fantasy", "year": 1937, "description": "A homebody hobbit is swept into an epic quest to reclaim a dwarven kingdom and its treasure from a dragon."},
        # Non-Fiction
        {"book_id": 21, "title": "Sapiens: A Brief History of Humankind", "author": "Yuval Noah Harari", "genre": "Non-Fiction", "year": 2011, "description": "An exploration of how Homo sapiens came to dominate the world through cognitive, agricultural, and scientific revolutions."},
        {"book_id": 22, "title": "Educated", "author": "Tara Westover", "genre": "Non-Fiction", "year": 2018, "description": "A memoir about a woman who grows up in a survivalist family and eventually earns a PhD from Cambridge University."},
        {"book_id": 23, "title": "Thinking, Fast and Slow", "author": "Daniel Kahneman", "genre": "Non-Fiction", "year": 2011, "description": "A Nobel laureate explores the two systems that drive the way we think: fast intuitive thinking and slow deliberate thinking."},
        {"book_id": 24, "title": "The Immortal Life of Henrietta Lacks", "author": "Rebecca Skloot", "genre": "Non-Fiction", "year": 2010, "description": "The story of Henrietta Lacks, whose cancer cells were taken without consent and became one of the most important tools in medicine."},
        {"book_id": 25, "title": "Atomic Habits", "author": "James Clear", "genre": "Non-Fiction", "year": 2018, "description": "A practical guide to building good habits and breaking bad ones using small behavioral changes and the science of habit formation."},
        # Romance
        {"book_id": 26, "title": "The Notebook", "author": "Nicholas Sparks", "genre": "Romance", "year": 1996, "description": "A sweeping love story about a couple separated by class differences who reunite years later with passion still burning."},
        {"book_id": 27, "title": "Outlander", "author": "Diana Gabaldon", "genre": "Romance", "year": 1991, "description": "A WWII nurse is transported back to 18th-century Scotland where she falls in love with a Highland warrior."},
        {"book_id": 28, "title": "Me Before You", "author": "Jojo Moyes", "genre": "Romance", "year": 2012, "description": "A young woman becomes a caretaker for a paralyzed man, and they form an unexpected bond that changes both their lives."},
        {"book_id": 29, "title": "The Fault in Our Stars", "author": "John Green", "genre": "Romance", "year": 2012, "description": "Two teenagers with cancer fall in love and embark on a journey to Amsterdam to meet their favorite author."},
        {"book_id": 30, "title": "Beach Read", "author": "Emily Henry", "genre": "Romance", "year": 2020, "description": "Two writers with opposite styles challenge each other to write in the other's genre during a summer at neighboring beach houses."},
        # Horror
        {"book_id": 31, "title": "The Shining", "author": "Stephen King", "genre": "Horror", "year": 1977, "description": "A family becomes caretakers of an isolated hotel for the winter where supernatural forces drive the father to madness."},
        {"book_id": 32, "title": "Dracula", "author": "Bram Stoker", "genre": "Horror", "year": 1897, "description": "The classic gothic novel about Count Dracula's attempt to move from Transylvania to England to spread the undead curse."},
        {"book_id": 33, "title": "It", "author": "Stephen King", "genre": "Horror", "year": 1986, "description": "A group of childhood friends reunite to confront an evil entity that takes the form of a terrifying clown."},
        {"book_id": 34, "title": "Mexican Gothic", "author": "Silvia Moreno-Garcia", "genre": "Horror", "year": 2020, "description": "A young socialite investigates her cousin's mysterious illness at a decaying English mansion in 1950s Mexico."},
        {"book_id": 35, "title": "The Haunting of Hill House", "author": "Shirley Jackson", "genre": "Horror", "year": 1959, "description": "Four people stay in a notoriously haunted house as part of a paranormal investigation, facing terrifying psychological horror."},
        # Self-Help
        {"book_id": 36, "title": "How to Win Friends and Influence People", "author": "Dale Carnegie", "genre": "Self-Help", "year": 1936, "description": "A timeless guide to improving social skills, building relationships, and influencing others through empathy and communication."},
        {"book_id": 37, "title": "The 7 Habits of Highly Effective People", "author": "Stephen Covey", "genre": "Self-Help", "year": 1989, "description": "A principle-centered approach to personal and professional effectiveness through developing proactive and synergistic habits."},
        {"book_id": 38, "title": "The Power of Now", "author": "Eckhart Tolle", "genre": "Self-Help", "year": 1997, "description": "A spiritual guide to living in the present moment and freeing yourself from the tyranny of the thinking mind."},
        {"book_id": 39, "title": "Grit: The Power of Passion and Perseverance", "author": "Angela Duckworth", "genre": "Self-Help", "year": 2016, "description": "Research showing that sustained passion and persistence matter more than talent in achieving long-term success."},
        {"book_id": 40, "title": "Mindset: The New Psychology of Success", "author": "Carol S. Dweck", "genre": "Self-Help", "year": 2006, "description": "Explores the difference between a fixed mindset and a growth mindset, and how changing your beliefs can change your life."},
        # History
        {"book_id": 41, "title": "Guns, Germs, and Steel", "author": "Jared Diamond", "genre": "History", "year": 1997, "description": "An examination of why certain civilizations rose to power, focusing on geography, agriculture, and technology as key factors."},
        {"book_id": 42, "title": "The Diary of a Young Girl", "author": "Anne Frank", "genre": "History", "year": 1947, "description": "The diary of a Jewish girl hiding from Nazi persecution in Amsterdam during World War II, showing resilience and hope."},
        {"book_id": 43, "title": "A People's History of the United States", "author": "Howard Zinn", "genre": "History", "year": 1980, "description": "American history told from the perspective of marginalized groups including Native Americans, slaves, and workers."},
        {"book_id": 44, "title": "The Silk Roads", "author": "Peter Frankopan", "genre": "History", "year": 2015, "description": "A fresh perspective on world history centered on the ancient trade routes connecting East and West."},
        {"book_id": 45, "title": "Homo Deus: A Brief History of Tomorrow", "author": "Yuval Noah Harari", "genre": "History", "year": 2015, "description": "A look at humanity's future as technology and biology converge, exploring questions of immortality and artificial intelligence."},
        # Science
        {"book_id": 46, "title": "A Brief History of Time", "author": "Stephen Hawking", "genre": "Science", "year": 1988, "description": "A landmark popular science book explaining cosmology, black holes, and the nature of time for general readers."},
        {"book_id": 47, "title": "The Gene: An Intimate History", "author": "Siddhartha Mukherjee", "genre": "Science", "year": 2016, "description": "The story of the gene from its discovery to cutting-edge genetic engineering, intertwining science with personal narrative."},
        {"book_id": 48, "title": "Cosmos", "author": "Carl Sagan", "genre": "Science", "year": 1980, "description": "An awe-inspiring journey through the universe exploring the origins of life, galaxies, and humanity's place in the cosmos."},
        {"book_id": 49, "title": "The Selfish Gene", "author": "Richard Dawkins", "genre": "Science", "year": 1976, "description": "A revolutionary look at evolution from the gene's perspective, introducing the concept of the meme and selfish genetic replication."},
        {"book_id": 50, "title": "Silent Spring", "author": "Rachel Carson", "genre": "Science", "year": 1962, "description": "A groundbreaking environmental science book documenting the harmful effects of pesticides on ecosystems and human health."},

        # ===== 50 NEW BOOKS (IDs 51-100) =====

        # Romance (15 new books)
        {"book_id": 51, "title": "It Ends with Us", "author": "Colleen Hoover", "genre": "Romance", "year": 2016, "description": "A young woman in a new relationship discovers unsettling parallels between her partner and her abusive father, forcing painful choices about love."},
        {"book_id": 52, "title": "The Hating Game", "author": "Sally Thorne", "genre": "Romance", "year": 2016, "description": "Two executive assistants who despise each other find their rivalry slowly transforming into an undeniable romantic attraction."},
        {"book_id": 53, "title": "People We Meet on Vacation", "author": "Emily Henry", "genre": "Romance", "year": 2021, "description": "Two best friends who take annual vacations together had a falling out two years ago, and one last trip might fix everything or ruin it forever."},
        {"book_id": 54, "title": "The Kiss Quotient", "author": "Helen Hoang", "genre": "Romance", "year": 2018, "description": "A woman with Asperger's hires a professional escort to teach her about intimacy, but real feelings soon complicate the arrangement."},
        {"book_id": 55, "title": "Red, White & Royal Blue", "author": "Casey McQuiston", "genre": "Romance", "year": 2019, "description": "The son of the US President and the Prince of Wales turn their public feud into a secret romance that could shake two nations."},
        {"book_id": 56, "title": "The Spanish Love Deception", "author": "Elena Armas", "genre": "Romance", "year": 2021, "description": "A woman asks her annoying coworker to pose as her boyfriend at a wedding in Spain, leading to unexpected sparks and genuine feelings."},
        {"book_id": 57, "title": "Book Lovers", "author": "Emily Henry", "genre": "Romance", "year": 2022, "description": "A cutthroat literary agent on vacation in a small town keeps running into a brooding book editor, and their sharp banter turns into something more."},
        {"book_id": 58, "title": "The Love Hypothesis", "author": "Ali Hazelwood", "genre": "Romance", "year": 2021, "description": "A PhD candidate fakes a relationship with a stern professor to convince her best friend she's moved on, but the pretense becomes dangerously real."},
        {"book_id": 59, "title": "Ugly Love", "author": "Colleen Hoover", "genre": "Romance", "year": 2014, "description": "A young nurse agrees to a no-strings-attached relationship with a brooding pilot, but emotions prove impossible to keep out of the arrangement."},
        {"book_id": 60, "title": "The Rosie Project", "author": "Graeme Simsion", "genre": "Romance", "year": 2013, "description": "A socially awkward genetics professor designs a questionnaire to find the perfect wife but falls for a woman who is completely wrong on paper."},
        {"book_id": 61, "title": "Wuthering Heights", "author": "Emily Bronte", "genre": "Romance", "year": 1847, "description": "A passionate and destructive love story between Heathcliff and Catherine on the wild Yorkshire moors spanning two generations."},
        {"book_id": 62, "title": "Jane Eyre", "author": "Charlotte Bronte", "genre": "Romance", "year": 1847, "description": "An orphaned governess falls in love with her mysterious employer Mr. Rochester, only to discover a terrible secret hidden in his mansion."},
        {"book_id": 63, "title": "Normal People", "author": "Sally Rooney", "genre": "Romance", "year": 2018, "description": "Two Irish teenagers from different social backgrounds navigate an intense on-and-off relationship through school and university years."},
        {"book_id": 64, "title": "The Time Traveler's Wife", "author": "Audrey Niffenegger", "genre": "Romance", "year": 2003, "description": "A love story between a man with a genetic disorder that causes him to time travel unpredictably and the woman who must cope with his absences."},
        {"book_id": 65, "title": "Call Me by Your Name", "author": "Andre Aciman", "genre": "Romance", "year": 2007, "description": "A seventeen-year-old boy falls deeply in love with a visiting scholar during a sun-drenched summer on the Italian Riviera."},

        # Fiction (5 new books)
        {"book_id": 66, "title": "The Kite Runner", "author": "Khaled Hosseini", "genre": "Fiction", "year": 2003, "description": "A haunting tale of friendship, betrayal, and redemption set against the backdrop of a changing Afghanistan from monarchy to Taliban rule."},
        {"book_id": 67, "title": "The Book Thief", "author": "Markus Zusak", "genre": "Fiction", "year": 2005, "description": "Narrated by Death, the story of a young girl in Nazi Germany who steals books and shares them with neighbors during bombing raids."},
        {"book_id": 68, "title": "Life of Pi", "author": "Yann Martel", "genre": "Fiction", "year": 2001, "description": "A young Indian boy survives 227 days stranded on a lifeboat in the Pacific Ocean with a Bengal tiger named Richard Parker."},
        {"book_id": 69, "title": "Beloved", "author": "Toni Morrison", "genre": "Fiction", "year": 1987, "description": "A former slave is haunted by the ghost of her dead daughter in this powerful exploration of the trauma and legacy of slavery."},
        {"book_id": 70, "title": "The Alchemist", "author": "Paulo Coelho", "genre": "Fiction", "year": 1988, "description": "A young Andalusian shepherd embarks on a journey to the Egyptian pyramids in search of treasure and discovers his personal legend."},

        # Science Fiction (5 new books)
        {"book_id": 71, "title": "The Martian", "author": "Andy Weir", "genre": "Science Fiction", "year": 2011, "description": "An astronaut stranded alone on Mars must use his ingenuity and wit to survive and signal Earth for rescue before his supplies run out."},
        {"book_id": 72, "title": "Brave New World", "author": "Aldous Huxley", "genre": "Science Fiction", "year": 1932, "description": "A dystopian society uses genetic engineering and conditioning to maintain social stability, until an outsider challenges its shallow happiness."},
        {"book_id": 73, "title": "Fahrenheit 451", "author": "Ray Bradbury", "genre": "Science Fiction", "year": 1953, "description": "In a future society where books are banned and firemen burn them, one fireman begins to question everything after meeting a curious girl."},
        {"book_id": 74, "title": "The Left Hand of Darkness", "author": "Ursula K. Le Guin", "genre": "Science Fiction", "year": 1969, "description": "An envoy from Earth visits a planet where inhabitants have no fixed gender, challenging assumptions about identity and society."},
        {"book_id": 75, "title": "Project Hail Mary", "author": "Andy Weir", "genre": "Science Fiction", "year": 2021, "description": "An amnesiac astronaut wakes up alone on a spaceship and must save Earth from an extinction-level threat with the help of an alien companion."},

        # Mystery (5 new books)
        {"book_id": 76, "title": "The Woman in the Window", "author": "A.J. Finn", "genre": "Mystery", "year": 2018, "description": "An agoraphobic woman spying on her neighbors believes she witnesses a crime but struggles to convince anyone of what she saw."},
        {"book_id": 77, "title": "Big Little Lies", "author": "Liane Moriarty", "genre": "Mystery", "year": 2014, "description": "Three women in a wealthy coastal town become entangled in a murder investigation that unravels secrets of bullying, affairs, and domestic violence."},
        {"book_id": 78, "title": "In the Woods", "author": "Tana French", "genre": "Mystery", "year": 2007, "description": "A detective investigates a child's murder in the same woods where his own childhood friends mysteriously disappeared twenty years ago."},
        {"book_id": 79, "title": "The Girl on the Train", "author": "Paula Hawkins", "genre": "Mystery", "year": 2015, "description": "A woman who watches a perfect couple from her commuter train window becomes entangled in their lives after witnessing something shocking."},
        {"book_id": 80, "title": "Sharp Objects", "author": "Gillian Flynn", "genre": "Mystery", "year": 2006, "description": "A journalist with a troubled past returns to her small hometown to cover the murders of two young girls and confronts her own dark history."},

        # Fantasy (5 new books)
        {"book_id": 81, "title": "Mistborn: The Final Empire", "author": "Brandon Sanderson", "genre": "Fantasy", "year": 2006, "description": "A street thief discovers she has rare magical powers and joins a crew of rebels plotting to overthrow an immortal tyrant ruler."},
        {"book_id": 82, "title": "The Way of Kings", "author": "Brandon Sanderson", "genre": "Fantasy", "year": 2010, "description": "A war-torn world faces an ancient threat as a slave soldier, a scholar, and a highborn warrior discover their intertwined destinies."},
        {"book_id": 83, "title": "Circe", "author": "Madeline Miller", "genre": "Fantasy", "year": 2018, "description": "The story of the witch Circe from Greek mythology, banished to a remote island, who discovers her own power through encounters with gods and mortals."},
        {"book_id": 84, "title": "The Priory of the Orange Tree", "author": "Samantha Shannon", "genre": "Fantasy", "year": 2019, "description": "A world divided between dragonriders and those who fear them must unite when an ancient evil stirs and threatens to consume everything."},
        {"book_id": 85, "title": "Piranesi", "author": "Susanna Clarke", "genre": "Fantasy", "year": 2020, "description": "A man lives in a vast mysterious house filled with endless halls and ocean tides, slowly uncovering the truth of who he is and how he got there."},

        # Horror (5 new books)
        {"book_id": 86, "title": "The Exorcist", "author": "William Peter Blatty", "genre": "Horror", "year": 1971, "description": "A mother seeks the help of two priests to save her daughter from a terrifying demonic possession in their Georgetown home."},
        {"book_id": 87, "title": "Bird Box", "author": "Josh Malerman", "genre": "Horror", "year": 2014, "description": "A mother and her children navigate a post-apocalyptic world blindfolded, where seeing mysterious creatures outside drives people to fatal violence."},
        {"book_id": 88, "title": "House of Leaves", "author": "Mark Z. Danielewski", "genre": "Horror", "year": 2000, "description": "A family discovers their house is bigger on the inside than the outside, leading to a labyrinthine descent into madness and darkness."},
        {"book_id": 89, "title": "The Turn of the Screw", "author": "Henry James", "genre": "Horror", "year": 1898, "description": "A governess becomes convinced that the two children in her care are being visited by malevolent ghosts of former servants."},
        {"book_id": 90, "title": "Pet Sematary", "author": "Stephen King", "genre": "Horror", "year": 1983, "description": "A family discovers a burial ground behind their new home that can bring the dead back to life, but what returns is never quite the same."},

        # Non-Fiction (3 new books)
        {"book_id": 91, "title": "Becoming", "author": "Michelle Obama", "genre": "Non-Fiction", "year": 2018, "description": "The memoir of the former First Lady of the United States, tracing her journey from the South Side of Chicago to the White House."},
        {"book_id": 92, "title": "The Body Keeps the Score", "author": "Bessel van der Kolk", "genre": "Non-Fiction", "year": 2014, "description": "A pioneering researcher reveals how trauma reshapes the body and brain, and offers new paths to recovery and healing."},
        {"book_id": 93, "title": "Outliers", "author": "Malcolm Gladwell", "genre": "Non-Fiction", "year": 2008, "description": "An investigation into the factors that contribute to high levels of success, from cultural legacies to the famous 10,000-hour rule."},

        # Self-Help (3 new books)
        {"book_id": 94, "title": "The Subtle Art of Not Giving a F*ck", "author": "Mark Manson", "genre": "Self-Help", "year": 2016, "description": "A counterintuitive approach to living a good life by choosing what to care about and embracing limitations and uncertainty."},
        {"book_id": 95, "title": "You Are a Badass", "author": "Jen Sincero", "genre": "Self-Help", "year": 2013, "description": "A humorous and inspiring guide to identifying and overcoming self-sabotaging beliefs and habits to create a life you love."},
        {"book_id": 96, "title": "Deep Work", "author": "Cal Newport", "genre": "Self-Help", "year": 2016, "description": "Rules for focused success in a distracted world, arguing that the ability to concentrate deeply is becoming increasingly rare and valuable."},

        # History (2 new books)
        {"book_id": 97, "title": "The Wright Brothers", "author": "David McCullough", "genre": "History", "year": 2015, "description": "The dramatic story of two bicycle mechanics from Ohio who changed the world by achieving the first powered flight at Kitty Hawk."},
        {"book_id": 98, "title": "Genghis Khan and the Making of the Modern World", "author": "Jack Weatherford", "genre": "History", "year": 2004, "description": "A reassessment of Genghis Khan as a visionary leader who connected East and West and shaped the modern world through trade and law."},

        # Science (2 new books)
        {"book_id": 99, "title": "Astrophysics for People in a Hurry", "author": "Neil deGrasse Tyson", "genre": "Science", "year": 2017, "description": "A concise and accessible tour of the universe covering dark matter, quarks, the Big Bang, and humanity's place in the cosmos."},
        {"book_id": 100, "title": "The Sixth Extinction", "author": "Elizabeth Kolbert", "genre": "Science", "year": 2014, "description": "An exploration of the current mass extinction event caused by human activity, weaving field reports with the history of life on Earth."},
    ]
    return pd.DataFrame(books)


def generate_ratings(num_users=10, seed=42):
    """
    Generate synthetic user ratings with realistic patterns.
    Users have genre preferences that influence their ratings.
    """
    np.random.seed(seed)
    books_df = generate_books()

    genre_list = books_df["genre"].unique()
    total_books = len(books_df)

    ratings = []
    for user_id in range(1, num_users + 1):
        # Each user has 1-3 preferred genres
        fav_genres = np.random.choice(genre_list, size=np.random.randint(1, 4), replace=False)

        # Each user rates between 12 and 50 books (more books in dataset now)
        num_ratings = np.random.randint(12, 51)
        rated_books = np.random.choice(books_df["book_id"].values, size=min(num_ratings, total_books), replace=False)

        for book_id in rated_books:
            book_genre = books_df.loc[books_df["book_id"] == book_id, "genre"].values[0]
            if book_genre in fav_genres:
                rating = np.clip(np.random.normal(4.2, 0.7), 1, 5)
            else:
                rating = np.clip(np.random.normal(2.8, 1.0), 1, 5)
            rating = round(rating * 2) / 2  # Round to nearest 0.5
            rating = max(1.0, min(5.0, rating))
            ratings.append({
                "user_id": user_id,
                "book_id": int(book_id),
                "rating": rating
            })

    return pd.DataFrame(ratings)


def load_or_generate_data():
    """Load data from CSV or generate if not present."""
    os.makedirs("data", exist_ok=True)

    # Always regenerate to pick up any new books added to generate_books()
    books_df = generate_books()
    ratings_df = generate_ratings()

    # Try to save to CSV (may fail on OneDrive-locked files, which is okay)
    try:
        books_df.to_csv(BOOKS_CSV, index=False)
        ratings_df.to_csv(RATINGS_CSV, index=False)
    except PermissionError:
        print("  Warning: Could not write CSV cache (file locked). Using in-memory data.")

    return books_df, ratings_df


if __name__ == "__main__":
    books, ratings = load_or_generate_data()
    print(f"Books: {len(books)}, Ratings: {len(ratings)}")
    print(f"Users: {ratings['user_id'].nunique()}")
    print(f"Avg ratings per user: {len(ratings) / ratings['user_id'].nunique():.1f}")
    print(f"\nGenre distribution:\n{books['genre'].value_counts()}")
    print(f"\nRating distribution:\n{ratings['rating'].value_counts().sort_index()}")
