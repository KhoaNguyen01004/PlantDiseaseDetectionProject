package com.example

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil.compose.AsyncImage

class GuideActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(
                colorScheme = lightColorScheme(
                    background = Color(0xFFFBF9F6),
                    surface = Color(0xFFEAE1D3),
                    primary = Color(0xFF324D3E),
                    secondary = Color(0xFFEEE8E0),
                    onBackground = Color(0xFF1D1B16),
                    onSurface = Color(0xFF1D1B16)
                )
            ) {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = Color(0xFFFBF9F6)
                ) {
                    GuideScreen(onBack = { finish() })
                }
            }
        }
    }
}

data class PlantGuideItem(
    val name: String,
    val category: String, // e.g. "Fruit", "Vegetable", "Grain"
    val imageUrl: String,
    val characteristics: List<String>,
    val vulnerableDiseases: List<String>
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GuideScreen(onBack: () -> Unit) {
    var searchQuery by remember { mutableStateOf("") }
    var selectedCategory by remember { mutableStateOf("All") }

    val plants = remember { getPlantGuides() }
    val categories = remember { listOf("All", "Fruit", "Vegetable", "Grain", "Herbal") }

    // Filter plants based on search and selected category
    val filteredPlants = remember(searchQuery, selectedCategory) {
        plants.filter { plant ->
            val matchesCategory = (selectedCategory == "All") || (plant.category == selectedCategory)
            val matchesSearch = plant.name.contains(searchQuery, ignoreCase = true) ||
                    plant.characteristics.any { it.contains(searchQuery, ignoreCase = true) } ||
                    plant.vulnerableDiseases.any { it.contains(searchQuery, ignoreCase = true) }
            matchesCategory && matchesSearch
        }
    }

    Column(modifier = Modifier.fillMaxSize()) {
        // App Bar
        TopAppBar(
            title = {
                Text(
                    text = "Plant Identification Guide",
                    fontWeight = FontWeight.Bold,
                    fontSize = 20.sp,
                    color = Color(0xFF1D1B16)
                )
            },
            navigationIcon = {
                IconButton(onClick = onBack) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back to Scanner",
                        tint = Color(0xFF524B3A)
                    )
                }
            },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = Color(0xFFFBF9F6)
            )
        )

        // Search Bar
        OutlinedTextField(
            value = searchQuery,
            onValueChange = { searchQuery = it },
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 8.dp),
            placeholder = { Text("Search plants, traits, diseases...", color = Color(0xFF8C8270)) },
            leadingIcon = {
                Icon(
                    imageVector = Icons.Default.Search,
                    contentDescription = "Search icon",
                    tint = Color(0xFF524B3A)
                )
            },
            shape = RoundedCornerShape(16.dp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedBorderColor = Color(0xFF324D3E),
                unfocusedBorderColor = Color(0xFFD7D0C5),
                focusedContainerColor = Color(0xFFF1EDE6),
                unfocusedContainerColor = Color(0xFFF1EDE6)
            ),
            singleLine = true
        )

        // Filter Category Chips
        LazyRow(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 4.dp, horizontal = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(categories) { category ->
                val isSelected = category == selectedCategory
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(100.dp))
                        .background(if (isSelected) Color(0xFF324D3E) else Color(0xFFEEE8E0))
                        .border(
                            width = 1.dp,
                            color = if (isSelected) Color(0xFF324D3E) else Color(0xFFD7D0C5),
                            shape = RoundedCornerShape(100.dp)
                        )
                        .clickable { selectedCategory = category }
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = category,
                        color = if (isSelected) Color.White else Color(0xFF524B3A),
                        fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
                        fontSize = 13.sp
                    )
                }
            }
        }

        // Plants List
        if (filteredPlants.isEmpty()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                contentAlignment = Alignment.Center
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "No plants found",
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF524B3A),
                        fontSize = 16.sp
                    )
                    Text(
                        text = "Try adjusting your search or category filter",
                        color = Color(0xFF8C8270),
                        fontSize = 14.sp,
                        modifier = Modifier.padding(top = 4.dp)
                    )
                }
            }
        } else {
            LazyColumn(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                items(filteredPlants) { plant ->
                    PlantCard(plant = plant)
                }
            }
        }
    }
}

@Composable
fun PlantCard(plant: PlantGuideItem) {
    val context = LocalContext.current
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .border(1.dp, Color(0xFFD7CDBE), RoundedCornerShape(24.dp)),
        shape = RoundedCornerShape(24.dp),
        colors = CardDefaults.cardColors(
            containerColor = Color(0xFFEAE1D3)
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            // Header Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = plant.name,
                    fontSize = 20.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color(0xFF1D1B16)
                )

                // Category tag
                Box(
                    modifier = Modifier
                        .background(Color(0xFF324D3E), RoundedCornerShape(100.dp))
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = plant.category,
                        color = Color.White,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Body Row: Image left, traits right
            Row(modifier = Modifier.fillMaxWidth()) {
                // Leaf Image (Async load with Coil)
                AsyncImage(
                    model = plant.imageUrl,
                    contentDescription = "${plant.name} guide image",
                    modifier = Modifier
                        .size(100.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .background(Color(0xFFDDD6C9))
                        .border(1.dp, Color(0xFFD7D0C5), RoundedCornerShape(16.dp)),
                    contentScale = ContentScale.Crop
                )

                Spacer(modifier = Modifier.width(16.dp))

                // Characteristics column
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Characteristics:",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF524B3A),
                        modifier = Modifier.padding(bottom = 2.dp)
                    )
                    plant.characteristics.forEach { trait ->
                        Text(
                            text = "• $trait",
                            fontSize = 12.sp,
                            color = Color(0xFF3E382A),
                            lineHeight = 16.sp
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Vulnerable Diseases Chips row
            Text(
                text = "Tracked Diseases in AgriLens Model:",
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold,
                color = Color(0xFF524B3A),
                modifier = Modifier.padding(bottom = 6.dp)
            )

            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                plant.vulnerableDiseases.forEach { disease ->
                    Box(
                        modifier = Modifier
                            .background(Color(0xFFECE6DB), RoundedCornerShape(100.dp))
                            .border(1.dp, Color(0xFFD7CDBE), RoundedCornerShape(100.dp))
                            .clickable {
                                // Click to show info or alert
                            }
                            .padding(horizontal = 10.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = disease,
                            fontSize = 10.sp,
                            color = Color(0xFF324D3E),
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun FlowRow(
    modifier: Modifier = Modifier,
    horizontalArrangement: Arrangement.Horizontal = Arrangement.Start,
    verticalArrangement: Arrangement.Vertical = Arrangement.Top,
    content: @Composable () -> Unit
) {
    // Standard compose flow-row replacement to ensure clean layout compile
    androidx.compose.foundation.layout.FlowRow(
        modifier = modifier,
        horizontalArrangement = horizontalArrangement,
        verticalArrangement = verticalArrangement
    ) {
        content()
    }
}

fun getPlantGuides(): List<PlantGuideItem> {
    return listOf(
        PlantGuideItem(
            name = "Apple",
            category = "Fruit",
            imageUrl = "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Temperate deciduous tree",
                "Needs winter chilling hours",
                "Full sun & well-drained loam"
            ),
            vulnerableDiseases = listOf("Apple scab", "Black rot", "Cedar apple rust")
        ),
        PlantGuideItem(
            name = "Blueberry",
            category = "Fruit",
            imageUrl = "https://images.unsplash.com/photo-1596547609652-9cf5d8d76921?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Perennial acid-loving shrub",
                "Needs pH soil range of 4.5-4.8",
                "Fibrous root system, shallow watering"
            ),
            vulnerableDiseases = listOf("Healthy (Reference)")
        ),
        PlantGuideItem(
            name = "Cherry",
            category = "Fruit",
            imageUrl = "https://images.unsplash.com/photo-1527661591475-527312dd65f5?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Spring blooming stone fruit",
                "High cold/frost air sensitivity",
                "Enjoys deep sandy-mud base"
            ),
            vulnerableDiseases = listOf("Powdery mildew")
        ),
        PlantGuideItem(
            name = "Corn (Maize)",
            category = "Grain",
            imageUrl = "https://images.unsplash.com/photo-1530595467537-0b5996c41f2d?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Tall rapid-growing annual grass",
                "Extremely high nitrogen demand",
                "Wind-pollinated stalk layout"
            ),
            vulnerableDiseases = listOf("Cercospora leaf spot", "Common rust", "Northern Leaf Blight")
        ),
        PlantGuideItem(
            name = "Grape",
            category = "Fruit",
            imageUrl = "https://images.unsplash.com/photo-1537640538966-79f369143f8f?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Woody climbing fruiting vine",
                "Requires robust vertical trellis support",
                "Enjoys dry warm microclimate dry summer"
            ),
            vulnerableDiseases = listOf("Black rot", "Esca (Black Measles)", "Isariopsis leaf spot")
        ),
        PlantGuideItem(
            name = "Orange (Citrus)",
            category = "Fruit",
            imageUrl = "https://images.unsplash.com/photo-1547514701-42782101795e?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Evergreen sub-tropical tree",
                "Frost-sensitive, high clay loam moisture",
                "Sustained year-round photosynthesis"
            ),
            vulnerableDiseases = listOf("Citrus greening (HLB)")
        ),
        PlantGuideItem(
            name = "Peach",
            category = "Fruit",
            imageUrl = "https://images.unsplash.com/photo-1595124253363-c596e74b314b?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Deciduous orchard stone fruit tree",
                "Demands high insect drainage sandy bed",
                "Sensitive to waterlogged root centers"
            ),
            vulnerableDiseases = listOf("Bacterial spot")
        ),
        PlantGuideItem(
            name = "Pepper (Bell)",
            category = "Vegetable",
            imageUrl = "https://images.unsplash.com/photo-1563565038-a441ef2421cb?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Short perennial warm shrub",
                "Prefers hot steady root temperatures",
                "Compact self-pollinating flowers"
            ),
            vulnerableDiseases = listOf("Bacterial spot")
        ),
        PlantGuideItem(
            name = "Potato",
            category = "Vegetable",
            imageUrl = "https://images.unsplash.com/photo-1518977676601-b53f82aba655?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Starchy underground tuber nightshade",
                "Grows best in hilled loose soil ridges",
                "Prefers cool moist climates"
            ),
            vulnerableDiseases = listOf("Early blight", "Late blight")
        ),
        PlantGuideItem(
            name = "Raspberry",
            category = "Fruit",
            imageUrl = "https://images.unsplash.com/photo-1534080391025-a7db5e13d46c?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Biennial canes, perennial fiber root",
                "Thrives in dense organic soils",
                "Enjoys regular summer mulching"
            ),
            vulnerableDiseases = listOf("Healthy (Reference)")
        ),
        PlantGuideItem(
            name = "Soybean",
            category = "Grain",
            imageUrl = "https://images.unsplash.com/photo-1599599810769-bcde5a160d32?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Nitrogen-fixing legume annual",
                "Crucial protein rich bean structure",
                "Highly beneficial for crop rotation rows"
            ),
            vulnerableDiseases = listOf("Healthy (Reference)")
        ),
        PlantGuideItem(
            name = "Squash",
            category = "Vegetable",
            imageUrl = "https://images.unsplash.com/photo-1508349937151-22b68b72d5b1?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Vining ground cover herbaceous annual",
                "Requires large spaces, rich soils",
                "Produces edible male and female blossoms"
            ),
            vulnerableDiseases = listOf("Powdery mildew")
        ),
        PlantGuideItem(
            name = "Strawberry",
            category = "Fruit",
            imageUrl = "https://images.unsplash.com/photo-1464965911861-746a04b4bca6?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Low stoloniferous runner herb",
                "Grows in well-drained sandy borders",
                "Very shallow water-retention roots"
            ),
            vulnerableDiseases = listOf("Leaf scorch")
        ),
        PlantGuideItem(
            name = "Tomato",
            category = "Vegetable",
            imageUrl = "https://images.unsplash.com/photo-1592924357228-91a4daadcfea?auto=format&fit=crop&q=80&w=200",
            characteristics = listOf(
                "Vining solanaceous nightshade vine",
                "Thrives with staking, warm damp bases",
                "Extremely susceptible to damp-air molds"
            ),
            vulnerableDiseases = listOf(
                "Bacterial spot", "Early blight", "Late blight", "Leaf Mold",
                "Septoria leaf spot", "Target Spot", "Spider mites", "TYLCV", "Mosaic virus"
            )
        )
    )
}
